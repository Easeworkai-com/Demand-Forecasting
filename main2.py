import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import os
from streamlit_app import query_db, reset_memory, create_session, get_sessions, display_chat
import requests

# Set page config
#st.set_page_config(layout="wide", page_title="Demand Forecasting")

# Initialize session state variables
if 'product_name' not in st.session_state:
    st.session_state['product_name'] = ''
if 'forecast_duration' not in st.session_state:
    st.session_state['forecast_duration'] = 1
if 'duration_unit' not in st.session_state:
    st.session_state['duration_unit'] = ''
if 'forecast_shown' not in st.session_state:
    st.session_state['forecast_shown'] = False
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = None
if 'safety_stock' not in st.session_state:
    st.session_state['safety_stock'] = 0
if 'reorder_point' not in st.session_state:
    st.session_state['reorder_point'] = 0
if 'performance_metrics' not in st.session_state:
    st.session_state['performance_metrics'] = None

# Load your data
@st.cache_data
def load_data():
    return pd.read_csv("stock.csv", parse_dates=['Date'])

df1 = load_data()

def forecast_product(product_name, forecast_duration, duration_unit, economic_index, raw_material_price):
    lead_time = 10  # Default lead time in days
    service_level = 0.95  # Default service level (95%)

    df = df1[df1["Product_Name"] == product_name].copy()
    
    # Prepare the dataframe with additional features
    df_prophet = df[['Date', 'Units_Sold', 'Unit_Price', 'Lead_Time_Days', 'On_Time_Delivery_Rate', 'Category', 'Supplier_ID']].copy()
    df_prophet['Date'] = pd.to_datetime(df_prophet['Date'])
    df_prophet = df_prophet.sort_values(by=['Date'])

    # Rename columns to match Prophet requirements
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Units_Sold': 'y'})

    # Label encode categorical variables
    encoder = LabelEncoder()
    df_prophet['Category'] = encoder.fit_transform(df_prophet['Category'])
    df_prophet['Supplier_ID'] = encoder.fit_transform(df_prophet['Supplier_ID'])

    # Handle outliers using IQR method
    Q1 = df_prophet['y'].quantile(0.25)
    Q3 = df_prophet['y'].quantile(0.75)
    IQR = Q3 - Q1
    df_prophet = df_prophet[(df_prophet['y'] >= Q1 - 1.5*IQR) & (df_prophet['y'] <= Q3 + 1.5*IQR)]

    # Initialize and fit the Prophet model with adjusted parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        changepoint_range=0.8
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add the additional regressors
    model.add_regressor('Unit_Price')
    model.add_regressor('Lead_Time_Days')
    model.add_regressor('On_Time_Delivery_Rate')
    feature_names = ['Category', 'Supplier_ID']

    # Add encoded categorical features
    for feature in feature_names:
        model.add_regressor(feature)

    model.fit(df_prophet)

    # Convert the forecast duration into days based on the unit selected
    if duration_unit == 'weeks':
        forecast_days = forecast_duration * 7
    elif duration_unit == 'months':
        forecast_days = forecast_duration * 30
    else:
        forecast_days = forecast_duration

    # Make future predictions for the next forecast_days
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add the additional features to the future dataframe
    last_known_features = df_prophet[['ds', 'Unit_Price', 'Lead_Time_Days', 'On_Time_Delivery_Rate'] + list(feature_names)].iloc[-1].to_dict()
    future_features = pd.DataFrame([last_known_features] * len(future))
    future_features['ds'] = future['ds']
    future = pd.concat([future, future_features.drop('ds', axis=1)], axis=1)

    # Fill NaN values
    for column in future.columns:
        if column != 'ds':
            future[column] = future[column].fillna(df_prophet[column].mean())

    forecast = model.predict(future)

    # Extract future predictions starting after the last date in the dataset
    last_date = df_prophet['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_date]

    # Calculate the reorder point and safety stock
    forecast_table = future_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Demand'})
    forecast_table['Forecasted Demand'] = forecast_table['Forecasted Demand'].round().astype(int)

    mean_demand_per_day = forecast_table['Forecasted Demand'].mean()
    std_demand_per_day = forecast_table['Forecasted Demand'].std()

    mean_demand_lead_time = mean_demand_per_day * lead_time
    std_demand_lead_time = np.sqrt((lead_time * std_demand_per_day**2) + (mean_demand_per_day**2 * 2**2))

    z_score = norm.ppf(service_level)
    safety_stock = int(z_score * std_demand_lead_time)
    reorder_point = int(mean_demand_lead_time + safety_stock)

    # Perform cross-validation
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    
    st.session_state['forecast_data'] = future_forecast
    st.session_state['safety_stock'] = safety_stock
    st.session_state['reorder_point'] = reorder_point
    st.session_state['performance_metrics'] = df_p
    st.session_state['forecast_shown'] = True

    return future_forecast, safety_stock, reorder_point, df_p


def handle_chat():
    API_URL = "http://127.0.0.1:8000"
    logo_url = "https://github.com/SouSingh/reccomder/blob/main/logo.png?raw=true"
    st.sidebar.header("Ask EaseAI")
 
    # Reset Conversation Memory button at the top of the sidebar
    if st.sidebar.button("Reset Memory", key="reset_memory"):
        if 'session_id' in st.session_state:
            result = reset_memory(st.session_state['session_id'])
            st.sidebar.write(result)
            st.session_state['history'] = []
            st.experimental_rerun()
        else:
            st.sidebar.write("Please select a session.")
 
    # Display existing sessions in a list format
    sessions = get_sessions()
    for idx, session in enumerate(sessions):
        if st.sidebar.button(session, key=f"session_{idx}"):
            st.session_state['session_id'] = session
            st.experimental_rerun()
 
    # Create a new session button
    if st.sidebar.button("Create New Session", key="create_new_session"):
        session_id, message = create_session()
        if session_id:
            st.session_state['session_id'] = session_id
            st.sidebar.success(f"Session created: {session_id}")
            st.experimental_rerun()
        else:
            st.sidebar.error(message)
 
    # Load chat history for the selected session
    history = []
    if 'session_id' in st.session_state:
        try:
            history_response = requests.get(f"{API_URL}/history/{st.session_state['session_id']}")
            history_response.raise_for_status()
            history = history_response.json().get("history", [])
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Failed to load history: {e}")
 
    input_container = st.sidebar.container()
    with input_container:
        with st.form(key="input_form", clear_on_submit=True):
            prompt = st.text_input("User prompt", "", placeholder="Type your question here...", label_visibility="collapsed")
            submit_button = st.form_submit_button(" ➤ ")
 
            if submit_button:
                if prompt and 'session_id' in st.session_state:
                    response_text, conversation_history = query_db(prompt, st.session_state['session_id'])
                    st.session_state['history'] = conversation_history  # Update session history
                    st.experimental_rerun()
                else:
                    st.write("Please enter a prompt and select a session.")
 
    # Display chat history
    if history:
        for entry in history:
            role = entry["role"]
            message = entry["message"]
            if role == "User":
                st.sidebar.markdown(f"**User:** {message}")
            elif role == "EaseAI":
                st.sidebar.markdown(f"**EaseAI:** {message}")

handle_chat()
# Main layout
st.title('Demand Forecasting')

# Modify the selection widgets to use and update session state
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    product_name = st.selectbox('Select Product Name:', 
                                [''] + list(df1['Product_Name'].unique()), 
                                index=0,
                                key='product_name')


with col2:
    duration_unit = st.selectbox('Forecast Duration:', 
                                 ['', 'days', 'weeks', 'months'], 
                                 index=0,
                                 key='duration_unit')
with col3:
    forecast_duration = st.number_input(f'Number of {duration_unit.capitalize() if duration_unit else "Units"}:', 
                                        min_value=1, 
                                        step=1, 
                                        value=None,
                                        placeholder="Enter number")



col4, col5 = st.columns(2)
with col4:
    country = st.selectbox("Select Country", ["", "USA", "Canada", "UK", "Germany"], 
                           index=0,
                           key='country')

with col5:
    state_options = {
        "USA": ["", "California", "Texas", "New York"],
        "Canada": ["", "Ontario", "Quebec", "British Columbia"],
        "UK": ["", "England", "Scotland", "Wales"],
        "Germany": ["", "Bavaria", "Berlin", "Hamburg"]
    }
    state = st.selectbox("Select State", state_options.get(st.session_state.country, [""]), 
                         index=0,
                         key='state')

col6, col7 = st.columns(2)
with col6:
    economic_index = st.checkbox('Economic Index', key='economic_index')
    
with col7:
    raw_material_price = st.checkbox('Raw Material Price Index', key='raw_material_price')

# The forecast button can remain the same
if st.button('Forecast', use_container_width=True):
    if st.session_state.product_name:
        forecast_product(st.session_state.product_name, 
                         st.session_state.forecast_duration, 
                         st.session_state.duration_unit, 
                         st.session_state.economic_index, 
                         st.session_state.raw_material_price)
    else:
        st.write("Please select a product name.")
# Display forecast results if shown
if st.session_state['forecast_shown']:
    future_forecast = st.session_state['forecast_data']

    # Use tabs to organize the content
    tabs = st.tabs(["Forecast Plot", "Forecast Table", "Safety Stock", "Reorder Point"])

    with tabs[0]:
        st.subheader('Forecast Plot')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_forecast['ds'], future_forecast['yhat'], label='Predicted', marker='o')
        ax.fill_between(future_forecast['ds'], 
                        future_forecast['yhat_lower'], 
                        future_forecast['yhat_upper'], 
                        color='k', alpha=0.2)
        ax.legend()
        ax.set_title(f'Forecasted Demand for {st.session_state["product_name"]} for the Next {st.session_state["forecast_duration"]} {st.session_state["duration_unit"].capitalize()}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Units Sold')
        st.pyplot(fig)

    with tabs[1]:
        st.subheader('Forecast Table')
        forecast_table = future_forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted Demand'})
        forecast_table['Forecasted Demand'] = forecast_table['Forecasted Demand'].round().astype(int)
        st.write(forecast_table.to_html(index=False), unsafe_allow_html=True)

    with tabs[2]:
        st.subheader('Safety Stock')
        st.write(f'Safety Stock for {st.session_state["product_name"]}: {st.session_state["safety_stock"]}')
        st.write("Safety stock is the extra inventory kept on hand to prevent stockouts due to uncertainties in supply and demand.")

    with tabs[3]:
        st.subheader('Reorder Point')
        st.write(f'Reorder Point for {st.session_state["product_name"]}: {st.session_state["reorder_point"]}')
        st.write("The reorder point is the inventory level at which a new order should be placed to replenish stock.")

    