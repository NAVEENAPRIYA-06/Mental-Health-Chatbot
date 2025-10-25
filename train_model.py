# train_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

# 1. Model Selection
# Using 'microsoft/DialoGPT-small' for a fast-to-train conversational model
MODEL_NAME = "microsoft/DialoGPT-small"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set pad_token_id to eos_token_id for open-ended generation tasks
# This is a common practice for GPT-style models in conversation.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"Model {MODEL_NAME} loaded successfully.")

# train_model.py (add this below the model loading code)

# 2. Empathetic Dialogue Script
# Each entry is a turn. The model learns to predict the next response.
# We use the tokenizer's special tokens to separate turns.
# <|endoftext|> is the separator.

dialogues = [
    "User: I'm feeling really stressed about work lately. My deadline is impossible to meet." + tokenizer.eos_token + 
    "Chatbot: I hear you. That sounds like a heavy burden to carry. It's completely understandable to feel overwhelmed right now.",

    "User: I just feel so alone. I don't know who to talk to." + tokenizer.eos_token + 
    "Chatbot: I'm here to listen. You are not alone in this, and your feelings are valid. What's on your mind?",

    "User: Nothing seems to be going right. I feel like a failure." + tokenizer.eos_token + 
    "Chatbot: That's a painful thought, and I'm sorry you're going through that. Remember that one setback doesn't define your entire worth.",

    "User: I wish things were better. I'm just tired all the time." + tokenizer.eos_token + 
    "Chatbot: It sounds like you need some rest. What is one small thing you could do today to be kind to yourself?",

    # Add more empathetic dialogues here for better fine-tuning...
]
# train_model.py (add this below the dialogues list)

# 3. Data Tokenization and Preparation

# Combine all dialogues into a single string for block-wise processing
text = "\n".join(dialogues)

# Tokenize the entire text
tokenized_data = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

# Convert to dataset format (simple for this small example)
class ConversationalDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        # Model training requires 'input_ids' and 'labels'. 
        # For Causal LMs like DialoGPT, the 'labels' are just the 'input_ids'.
        self.input_ids = tokenized_data['input_ids'].squeeze()
        self.attention_mask = tokenized_data['attention_mask'].squeeze()
        self.labels = self.input_ids.clone() 

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# For a tiny dataset like this, we need to artificially expand it for the trainer
# In a real project, this would be handled by a proper dataset loader
inputs = [{'input_ids': tokenized_data['input_ids'].squeeze(), 
           'attention_mask': tokenized_data['attention_mask'].squeeze(), 
           'labels': tokenized_data['input_ids'].squeeze().clone()}]

# Repeat the single-entry list to mimic a small training batch/set
training_data = inputs * 100 

# We will use the simplest form of dataset input for the Trainer
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        # i will be a dictionary of tensors, which is what the trainer expects
        return self.encodings[i] 

train_dataset = SimpleDataset(training_data)

print("Data prepared for fine-tuning.")
# train_model.py (add this below the data preparation)

# 4. Fine-Tuning Configuration and Execution
output_dir = './dialo_model_finetuned'

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,              # Number of training epochs (3-5 is usually enough)
    per_device_train_batch_size=1,   # Batch size (set to 1 for small size)
    save_steps=1000,                 # Save checkpoint every X steps (not strictly needed here)
    logging_steps=50,                # Log results every X steps
    do_train=True,
    report_to="none"                 # Disable online reporting for simplicity
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting fine-tuning...")
trainer.train()

# 5. Save the Fine-tuned Model and Tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nFine-tuning complete. Model saved to: {output_dir}")