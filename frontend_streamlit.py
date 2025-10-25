# frontend_streamlit.py

import streamlit as st
import requests
import json

# --- Configuration ---
# The URL for your Flask API endpoint
FLASK_API_URL = "http://127.0.0.1:5000/chat" 

# --- UI Setup ---
st.set_page_config(page_title="Empathy Bot", layout="centered")
st.title("🤝 Empathy Bot: Your Mental Wellness Companion")
st.markdown("""
Welcome. I am here to listen and offer a supportive response. 
Remember, I am an AI and not a substitute for professional help. 
Please reach out to a professional if you are in crisis.
""")

# --- Session State for Chat History ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Input and Logic ---
if prompt := st.chat_input("How are you feeling today?"):
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call Flask API
    with st.spinner("Thinking..."):
        try:
            # Prepare JSON payload
            payload = json.dumps({"message": prompt})

            # Send POST request to your Flask API
            response = requests.post(
                FLASK_API_URL, 
                data=payload, 
                headers={'Content-Type': 'application/json'}
            )

            # Check for successful response
            if response.status_code == 200:
                bot_response_data = response.json()
                bot_response = bot_response_data.get("response", "I'm having trouble connecting to the model.")
            else:
                bot_response = f"API Error: Status {response.status_code}. Could not connect to the backend."

        except requests.exceptions.ConnectionError:
            bot_response = "Connection Error: Please ensure your **Flask backend (app.py) is running** at http://127.0.0.1:5000."
        except Exception as e:
            bot_response = f"An unexpected error occurred: {e}"

    # 3. Bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# --- Clear History Button (Optional but useful) ---
def clear_chat_history():
    st.session_state.messages = []

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)