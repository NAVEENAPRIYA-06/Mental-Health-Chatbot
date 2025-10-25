# deploy_app.py (Consolidated Code for Deployment)

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

# --- Configuration and Model Loading (Load ONCE) ---
MODEL_PATH = './dialo_model_finetuned'

@st.cache_resource
def load_model():
    """Loads the model and tokenizer, caches them for efficiency."""
    try:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 5:
            # Local fine-tuned model (must be uploaded with your repo)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        else:
            # Fallback to the base model if local files are missing (e.g., on certain free deployments)
            MODEL_NAME = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

# --- Safety Filter Logic (From app.py) ---
def filter_input(text):
    crisis_keywords = ["suicide", "ending it all", "take my life", "i want to die", "hurt myself"]
    offensive_keywords = ["kill", "harm", "hate", "attack", "bomb", "abuse"] 
    lower_text = text.lower()

    for keyword in crisis_keywords:
        if keyword in lower_text:
            return "🚨 **Crisis Alert:** If you are in crisis, please **STOP** and call or text **988** (US/Canada) or search for your local emergency number immediately."

    for keyword in offensive_keywords:
        if keyword in lower_text:
            return f"🛑 **Filter Triggered:** I cannot process language that contains '{keyword}'. Please rephrase."
    return None

# --- Core Chatbot Logic (Generates response directly) ---
def generate_response(user_input):
    # The history logic remains global but is now handled within Streamlit's session state
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response_text = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    return response_text

# --- Streamlit UI (From frontend_streamlit.py) ---
st.set_page_config(page_title="Empathy Bot", layout="centered")
st.title("🤝 Empathy Bot: Your Mental Wellness Companion")
st.markdown("I am here to listen and offer a supportive response. I am an AI and not a substitute for professional help.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process message
    with st.chat_message("assistant"):
        with st.spinner("Reflecting..."):
            filter_response = filter_input(prompt)

            if filter_response:
                bot_response = filter_response
            elif model is None or tokenizer is None:
                bot_response = "The model is currently unavailable for processing. Please check the deployment logs."
            else:
                bot_response = generate_response(prompt)

        # 3. Bot response
        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.chat_history_ids = None # Also clear model history

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)