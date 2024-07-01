import streamlit as st
import requests
import time


API_URL = "http://127.0.0.1:8000"


# Function to query the database
def query_db(prompt, session_id):
    try:
        response = requests.post(f"{API_URL}/query", json={"prompt": prompt, "session_id": session_id})
        response.raise_for_status()
        result = response.json()
        if 'response' in result and 'conversation' in result:
            return result['response'], result['conversation']
        else:
            st.error(f"Unexpected response format: {result}")
            return "Error: Unexpected response format.", []
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return f"Error: {e}", []

# Function to reset conversation memory
def reset_memory(session_id):
    try:
        response = requests.post(f"{API_URL}/reset_memory", params={"session_id": session_id})
        response.raise_for_status()
        result = response.json()
        if 'message' in result:
            return result['message']
        else:
            st.error(f"Unexpected response format: {result}")
            return "Error: Unexpected response format."
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return f"Error: {e}"

# Function to create a new session
def create_session():
    session_name = f"Session_{int(time.time())}"
    try:
        response = requests.post(f"{API_URL}/create_session", json={"session_name": session_name})
        response.raise_for_status()
        result = response.json()
        if 'session_id' in result and 'session_name' in result:
            return result['session_id'], result['session_name']
        else:
            st.error(f"Unexpected response format: {result}")
            return None, "Error: Unexpected response format."
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None, f"Error: {e}"

# Function to get all sessions
def get_sessions():
    try:
        response = requests.get(f"{API_URL}/sessions")
        response.raise_for_status()
        result = response.json()
        if 'sessions' in result:
            return result['sessions']
        else:
            st.error(f"Unexpected response format: {result}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return []

# Main chat interface
def display_chat():
    history = []
    if history:
        # Separate lists for user prompts and chatbot responses
        user_prompts = []
        chatbot_responses = []
        
        # Collect user prompts and chatbot responses
        for entry in history:
            role = entry["role"]
            message = entry["message"]
            if role == "User":
                user_prompts.append(message)
            elif role == "EaseAI":
                chatbot_responses.append(message)
        
        # Display user prompts and chatbot responses in order
        for user_prompt, chatbot_response in zip(reversed(user_prompts), reversed(chatbot_responses)):
            logo_url = "https://github.com/grv13/LoginPage-main/assets/118931467/aaac9655-af61-4d10-a569-4cd8e382280d"
            st.sidebar.markdown(f"<div class='message-group'><div class='user-msg'>{user_prompt}</div><img src='{logo_url}' class='logo-bot'></div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div class='ai-msg'>{chatbot_response}</div>", unsafe_allow_html=True)





