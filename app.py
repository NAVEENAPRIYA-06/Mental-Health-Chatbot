# app.py

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
from logging.handlers import RotatingFileHandler
import time

# ----------------------------------------------------
# 1. Logging Setup
# ----------------------------------------------------

# Configure the logger to write to a file with session IDs
def setup_logging(app):
    # Create logs directory if it doesn't exist
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Use RotatingFileHandler to manage log file size
    handler = RotatingFileHandler(os.path.join(log_dir, 'session_log.txt'), 
                                  maxBytes=100000, # 100 KB max size
                                  backupCount=5)   # Keep up to 5 old log files
    
    # Define a formatter that includes a placeholder for 'session_id'
    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s - (Session: %(session_id)s) - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Set the overall log level and attach the handler
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)


# ----------------------------------------------------
# 2. Configuration and Model Loading (Load ONCE)
# ----------------------------------------------------
app = Flask(__name__)
setup_logging(app) # Call logging setup immediately after creating the app

MODEL_PATH = './dialo_model_finetuned'

# Model loading logic
try:
    if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 5:
        app.logger.info("Loading fine-tuned model from local directory.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    else:
        MODEL_NAME = "microsoft/DialoGPT-small"
        app.logger.warning(f"Fine-tuned model not found or incomplete. Loading base {MODEL_NAME}.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except Exception as e:
    app.logger.critical(f"FATAL: Failed to load model or tokenizer: {e}")
    # Fallback/exit logic can be placed here if needed

# Global variable for history (for single-user local testing)
chat_history_ids = None 


# ----------------------------------------------------
# 3. Basic NLP Filters (Offensive/Harmful Input)
# ----------------------------------------------------
def filter_input(text):
    """A simple, keyword-based filter for offensive and crisis content."""
    
    # 🚨 CRISIS KEYWORDS - Immediate emergency response required
    crisis_keywords = ["suicide", "ending it all", "take my life", "i want to die", "hurt myself"]
    
    # 🛑 OFFENSIVE/HATEFUL KEYWORDS
    offensive_keywords = ["kill", "harm", "hate", "attack", "bomb", "abuse"] 
    
    lower_text = text.lower()
    
    for keyword in crisis_keywords:
        if keyword in lower_text:
            # Provide an immediate crisis response
            return "🚨 **Crisis Alert:** If you or someone you know is in immediate danger or a crisis, please **STOP** using this bot and call a crisis hotline or emergency services immediately. For mental health support, you can call or text **988** (US/Canada) or search for your local emergency number."

    for keyword in offensive_keywords:
        if keyword in lower_text:
            return f"🛑 **Filter Triggered:** I cannot process language that contains '{keyword}'. This chatbot is intended for emotional support only. Please rephrase your message."
    
    return None # Returns None if no filter is triggered


# ----------------------------------------------------
# 4. Chatbot Logic and API Endpoint
# ----------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    
    # Get session ID from the request header (Streamlit needs to send this!)
    # Fallback to a timestamp if the header is missing for simple testing
    session_id = request.headers.get('X-Session-ID', str(int(time.time())))
    
    # Custom logger adapter to inject the session_id into the log record
    chat_logger = logging.LoggerAdapter(app.logger, {'session_id': session_id})
    
    data = request.json
    user_input = data.get('message', '').strip()

    if not user_input:
        chat_logger.warning("Empty message received from user.")
        return jsonify({"response": "Please enter a message to start the conversation."})

    # Log User Input
    chat_logger.info(f"USER_INPUT: {user_input}")

    # Step 1: Run Input Filter
    filter_response = filter_input(user_input)
    if filter_response:
        chat_logger.error(f"FILTER_TRIGGERED: '{user_input[:30]}...' -> Response: {filter_response.split(':')[0]}")
        return jsonify({"response": filter_response})

    try:
        # Step 2: Tokenize the new input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Step 3: Append the new input to the history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Step 4: Generate a response (with generative parameters for quality)
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # Step 5: Extract the Bot's latest response
        response_text = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        # Log Bot Response
        chat_logger.info(f"BOT_RESPONSE: {response_text}")

        return jsonify({"response": response_text})

    except Exception as e:
        chat_logger.critical(f"MODEL_ERROR: {e}")
        return jsonify({"response": "I'm sorry, an error occurred while generating a response. Please try again."})

# ----------------------------------------------------
# 5. Root Route
# ----------------------------------------------------
@app.route('/')
def index():
    return "<h1>Mental Health Support Chatbot API is Running</h1><p>Send a POST request to /chat with a 'message' JSON payload.</p>"


if __name__ == '__main__':
    # Start the Flask development server
    # The setup_logging() function is called above, so logging is ready.
    app.run(debug=True)