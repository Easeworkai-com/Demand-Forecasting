import streamlit as st
from chat import (
    query_db,
    reset_memory,
    create_session,
    get_sessions,
    display_chat,
)
import requests



def handle_chat():
    API_URL = "http://127.0.0.1:8000"

    st.sidebar.header("Ask EaseAI")


    if st.sidebar.button("Reset Memory", key="reset_memory"):
        if "session_id" in st.session_state:
            result = reset_memory(st.session_state["session_id"])
            st.sidebar.write(result)
            st.session_state["history"] = []
            st.experimental_rerun()
        else:
            st.sidebar.write("Please select a session.")

    # Display existing sessions in a list format
    sessions = get_sessions()
    for idx, session in enumerate(sessions):
        if st.sidebar.button(session, key=f"session_{idx}"):
            st.session_state["session_id"] = session
            st.experimental_rerun()

    # Create a new session button
    if st.sidebar.button("Create New Session", key="create_new_session"):
        session_id, message = create_session()
        if session_id:
            st.session_state["session_id"] = session_id
            st.sidebar.success(f"Session created: {session_id}")
            st.experimental_rerun()
        else:
            st.sidebar.error(message)

    # Load chat history for the selected session
    history = []
    if "session_id" in st.session_state:
        try:
            history_response = requests.get(
                f"{API_URL}/history/{st.session_state['session_id']}"
            )
            history_response.raise_for_status()
            history = history_response.json().get("history", [])
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Failed to load history: {e}")

    input_container = st.sidebar.container()
    with input_container:
        with st.form(key="input_form", clear_on_submit=True):
            prompt = st.text_input(
                "User prompt",
                "",
                placeholder="Type your question here...",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(" ➤ ")

            if submit_button:
                if prompt and "session_id" in st.session_state:
                    response_text, conversation_history = query_db(
                        prompt, st.session_state["session_id"]
                    )
                    st.session_state["history"] = (
                        conversation_history  # Update session history
                    )
                    st.experimental_rerun()
                else:
                    st.write("Please enter a prompt and select a session.")

    # Display chat history
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

        for user_prompt, chatbot_response in zip(
            reversed(user_prompts), reversed(chatbot_responses)
        ):
            st.write('Hello')
            logo_url = ""
            st.sidebar.markdown(
                f"<img src='{logo_url}' class='logo-bot'>", unsafe_allow_html=True
            )
            st.sidebar.markdown(
                f"<img src='{logo_url}' class='logo-bot'><div class='user-msg'>{user_prompt}</div>", unsafe_allow_html=True
            )
            st.sidebar.markdown(
                f"<div class='ai-msg'>{chatbot_response}</div>", unsafe_allow_html=True
            )

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
