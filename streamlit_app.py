import streamlit as st
import requests
import time
# Set page config
st.set_page_config(layout="wide", page_title="Demand Forecasting")

API_URL = "http://127.0.0.1:8000"

# Custom CSS for chat-like interface
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #ffffff; /* White background */
    }
    .stApp {
        background-color: #ffffff; /* White background */
    }
    .main-content {
        flex: 1;
        overflow-y: auto; /* Scrollable content */
        padding-bottom: 60px; /* Space for the fixed footer */
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff; 
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.2);
        color: #000000;
    }
    .title {
        color: #5e17eb; /* Purple title */
        text-align: left;
        font-size: 30px;
        margin-top: 20px; /* Adjust margin-top as needed */
        margin-bottom: 10px; /* Adjust margin-bottom as needed */
    }
    .prompt-input {
        width: 100%;
        border: 1px solid #5e17eb; /* Purple border */
        border-radius: 10px;
        padding: 10px;
        color: #5e17eb; /* Purple text */
        background-color: #f3e9ff; /* Light purple background */
    }
    .prompt-input::placeholder {
        color: #5e17eb; /* Purple color for placeholder text */
    }
    .submit-button {
        background-color: #5e17eb; /* Purple */
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 10px;
    }
    .submit-button:hover {
        background-color: #4b13ba; /* Darker purple */
    }
    .reset-button {
        background-color: #5e17eb; /* Purple */
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 10px;
    }
    .reset-button:hover {
        background-color: #4b13ba; /* Darker purple */
    }
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px; /* Adjust the width as needed */
        height: auto;
        margin-bottom: 20px; /* Adjust margin-bottom as needed */
        border: 2px solid #5e17eb; /* Border around the logo */
        border-radius: 20px; /* Rounded corners */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Shadow effect */
    }
    .logo-bot{
        display: block;
        margin-left: 10px;
        margin-right: auto;
        width: 25px; /* Adjust the width as needed */
        height: 25px;
        margin-bottom: auto; /* Adjust margin-bottom as needed */
    }
    .input-text {
        color: #5e17eb; /* Purple text */
        font-size: 18px;
        margin-top: 5px; /* Adjust margin-top as needed */
        margin-bottom: 5px; /* Adjust margin-bottom as needed */
    }
    .result-text {
        background-color: #e5dff7; /* Light purple background */
        color: #5e17eb; /* Purple text */
        padding: 10px;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 10px;
    }
    .colored-label {
        color: #5e17eb; /* Purple color for label */
        font-size: 16px;
        margin-bottom: 5px;
    }
    .user-msg {
        background-color: #bab5f6; /* Light purple background */
        padding: 10px;
        border-radius: 10px;
        text-align: right;
        margin-bottom: 10px;
        width: 70%;
        margin-left: auto;
        color: #5e17eb; /* Purple text */
        font-weight: 550;
    }
    .ai-msg {
        background-color: #e7eaf6; /* Light purple background */
        padding: 10px;
        border-radius: 10px;
        text-align: left;
        margin-bottom: 10px;
        width: 70%;
        color: #5e17eb; /* Purple text */
    }
    
    .session-button {
        background-color: #5e17eb; /* Purple */
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 5px;
        text-align: left;
        width: 100%;
    }
    .session-button:hover {
        background-color: #4b13ba; /* Darker purple */
    }
    </style>
    """,
    unsafe_allow_html=True
)



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
            st.sidebar.markdown(f"<div class='user-msg'>{user_prompt}</div>", unsafe_allow_html=True)
            logo_url = "https://github.com/grv13/LoginPage-main/assets/118931467/aaac9655-af61-4d10-a569-4cd8e382280d"
            st.sidebar.markdown(f"<img src='{logo_url}' class='logo-bot'>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<div class='ai-msg'>{chatbot_response}</div>", unsafe_allow_html=True)





