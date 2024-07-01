import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import os
import streamlit_shadcn_ui as ui
import matplotlib.dates as mdates
from chat import (
    query_db,
    reset_memory,
    create_session,
    get_sessions,
    display_chat,
)
import requests
from src.components.add_logo import add_logo
from src.components.forecast_product import forecast_product
from load_data import df1

# Set page config
st.set_page_config(
    page_title="Demand Forecasting", page_icon="src/image/favicon.png", layout="wide"
)

with open("./src/css/styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


logo_url = "./logo.png"

st.markdown(f"<img src='logo.png' class='logo' />", unsafe_allow_html=True)


# Initialize session state variables
if "product_name" not in st.session_state:
    st.session_state["product_name"] = ""
if "forecast_duration" not in st.session_state:
    st.session_state["forecast_duration"] = None
if "duration_unit" not in st.session_state:
    st.session_state["duration_unit"] = ""
if "forecast_shown" not in st.session_state:
    st.session_state["forecast_shown"] = False
if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = None
if "safety_stock" not in st.session_state:
    st.session_state["safety_stock"] = 0
if "reorder_point" not in st.session_state:
    st.session_state["reorder_point"] = 0


def handle_chat():
    API_URL = "http://127.0.0.1:8000"

    st.sidebar.title("Ask EaseAI")

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
        prompt = st.chat_input("Type your question here...", key="chat_input")

        if prompt:  # This block will execute after the user submits a prompt.
            if "session_id" in st.session_state:
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
            user_pfp = (
                "https://cdn-icons-png.freepik.com/256/1077/1077114.png?semt=ais_hybrid"
            )
            logo_url = "https://github.com/grv13/LoginPage-main/assets/118931467/aaac9655-af61-4d10-a569-4cd8e382280d"
            st.sidebar.markdown(
                f"<div class='user-prompt'><div class='user-msg'>{user_prompt}</div><img src='{user_pfp}' class='user-pfp'></div>",
                unsafe_allow_html=True,
            )
            st.sidebar.markdown(
                f"<div class='ai-prompt'><img src='{logo_url}' class='logo-bot'><div class='ai-msg'>{chatbot_response}</div></div>",
                unsafe_allow_html=True,
            )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


handle_chat()


st.title("Demand Forecasting")
st.subheader("Forecasting Parameters")

# Modify the selection widgets to use and update session state
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    product_name = st.selectbox(
        "Select Product Name:",
        [""] + list(df1["Product_Name"].unique()),
        index=0,
        key="product_name",
    )


with col2:
    duration_unit = st.selectbox(
        "Forecast Duration:",
        ["", "days", "weeks", "months"],
        index=0,
        key="duration_unit",
    )
with col3:
    forecast_duration = st.number_input(
        f'Number of {st.session_state.duration_unit.capitalize() if st.session_state.duration_unit else "Units"}:',
        min_value=1,
        step=1,
        key="forecast_duration",
    )


col4, col5 = st.columns(2)
with col4:
    country = st.selectbox(
        "Select Country", ["", "USA", "Canada", "UK", "Germany"], index=0, key="country"
    )

with col5:
    state_options = {
        "USA": ["", "California", "Texas", "New York"],
        "Canada": ["", "Ontario", "Quebec", "British Columbia"],
        "UK": ["", "England", "Scotland", "Wales"],
        "Germany": ["", "Bavaria", "Berlin", "Hamburg"],
    }
    state = st.selectbox(
        "Select State",
        state_options.get(st.session_state.country, [""]),
        index=0,
        key="state",
    )

col6, col7, col8, col9 = st.columns(4)
with col6:
    economic_index = st.checkbox("Economic Index", key="economic_index")

with col7:
    raw_material_price = st.checkbox(
        "Raw Material Price Index", key="raw_material_price"
    )

with col8:
    Holidays = st.checkbox("Holidays", key="Holidays")

with col9:
    Promotions = st.checkbox("Promotions", key="Promotions")

col10, col11, col12, col13 = st.columns(4)
with col10:
    New_Product_Launches = st.checkbox(
        "New Product Launches", key="New_Product_Launches"
    )

with col11:
    Regulatory_Changes = st.checkbox("Regulatory Changes", key="Regulatory_Changes")

with col12:
    Supply_Chain_Disruptions = st.checkbox(
        "Supply Chain Disruptions", key="Supply_Chain_Disruptions"
    )

with col13:
    Demographic_Changes = st.checkbox("Demographic Changes", key="Demographic_Changes")


if ui.button(
    "Forecast",
    key="forecast_button",
    className="bg-violet-600 text-white",
):
    if (
        st.session_state.product_name
        and st.session_state.duration_unit
        and st.session_state.forecast_duration
    ):
        forecast_product(
            st.session_state.product_name,
            int(st.session_state.forecast_duration),  # Ensure this is an integer
            st.session_state.duration_unit,
            st.session_state.economic_index,
            st.session_state.raw_material_price,
            st.session_state.Holidays,
            st.session_state.Promotions,
            st.session_state.New_Product_Launches,
            st.session_state.Regulatory_Changes,
            st.session_state.Supply_Chain_Disruptions,
            st.session_state.Demographic_Changes,
        )
    else:
        st.write("Please select a product name, forecast duration, and duration unit.")


if st.session_state["forecast_shown"]:
    future_forecast = st.session_state["forecast_data"]

    # Use tabs to organize the content
    tabs = st.tabs(["Forecast Plot", "Forecast Table", "Safety Stock", "Reorder Point"])

    with tabs[0]:
        st.subheader("Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            future_forecast["ds"],
            future_forecast["yhat"],
            label="Predicted",
            marker="o",
            linestyle="-",
            color="b",
        )

        # Add labels to each point with improved positioning
        for i, row in future_forecast.iterrows():
            if (
                i == 0
                or i == len(future_forecast) - 1
                or i % (len(future_forecast) // 4) == 0
            ):
                ax.annotate(
                    f'{int(row["yhat"])}',
                    xy=(row["ds"], row["yhat"]),
                    textcoords="offset points",
                    xytext=(0, 10),  # Offset label by 10 points in y-direction
                    ha="center",
                    fontsize=10,  # Increased font size
                    color="blue",
                )

        ax.legend()
        ax.set_title(
            f'Forecasted Demand for {st.session_state["product_name"]} for the Next {st.session_state["forecast_duration"]} {st.session_state["duration_unit"].capitalize()}'
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")

        # Set custom ticks for x-axis
        date_range = future_forecast["ds"]
        ticks = pd.date_range(start=date_range.min(), end=date_range.max(), periods=7)
        ax.set_xticks(ticks)
        ax.set_xlim(date_range.min(), date_range.max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        # Add padding to y-axis limits to avoid label merging with the border
        y_min, y_max = future_forecast["yhat"].min(), future_forecast["yhat"].max()
        y_padding = (y_max - y_min) * 0.1  # Add 10% padding to top and bottom
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Adjust x-axis limits to avoid labels on vertical lines
        x_min, x_max = date_range.min(), date_range.max()
        ax.set_xlim(x_min - pd.Timedelta(days=1), x_max + pd.Timedelta(days=1))

        plt.xticks(rotation=45, ha="right")
        ax.figure.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Forecast Table")
        forecast_table = future_forecast[["ds", "yhat"]].rename(
            columns={"ds": "Date", "yhat": "Forecasted Demand"}
        )
        forecast_table["Forecasted Demand"] = (
            forecast_table["Forecasted Demand"].round().astype(int)
        )
        st.write(forecast_table.to_html(index=False), unsafe_allow_html=True)

    with tabs[2]:
        st.subheader("Safety Stock")
        st.write(
            f'Safety Stock for {st.session_state["product_name"]}: {st.session_state["safety_stock"]}'
        )
        st.write(
            "Safety stock is the extra inventory kept on hand to prevent stockouts due to uncertainties in supply and demand."
        )

    with tabs[3]:
        st.subheader("Reorder Point")
        st.write(
            f'Reorder Point for {st.session_state["product_name"]}: {st.session_state["reorder_point"]}'
        )
        st.write(
            "The reorder point is the inventory level at which a new order should be placed to replenish stock."
        )
