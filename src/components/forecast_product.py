import pandas as pd
import os
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics



def load_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Build the path to the stock.csv file
    stock_file_path = os.path.join(script_dir, '..', '..', 'data', 'stock.csv')
    return pd.read_csv(stock_file_path, parse_dates=["Date"])
df1 = load_data()


def forecast_product(
    product_name,
    forecast_duration,
    duration_unit,
    economic_index,
    raw_material_price,
    Holidays,
    Promotions,
    New_Product_Launches,
    Regulatory_Changes,
    Supply_Chain_Disruptions,
    Demographic_Changes,
):
    lead_time = 10  # Default lead time in days
    service_level = 0.95  # Default service level (95%)

    df = df1[df1["Product_Name"] == product_name].copy()

    # Prepare the dataframe with additional features
    df_prophet = df[
        [
            "Date",
            "Units_Sold",
            "Unit_Price",
            "Lead_Time_Days",
            "On_Time_Delivery_Rate",
            "Category",
            "Supplier_ID",
        ]
    ].copy()
    df_prophet["Date"] = pd.to_datetime(df_prophet["Date"])
    df_prophet = df_prophet.sort_values(by=["Date"])

    # Rename columns to match Prophet requirements
    df_prophet = df_prophet.rename(columns={"Date": "ds", "Units_Sold": "y"})

    # Label encode categorical variables
    encoder = LabelEncoder()
    df_prophet["Category"] = encoder.fit_transform(df_prophet["Category"])
    df_prophet["Supplier_ID"] = encoder.fit_transform(df_prophet["Supplier_ID"])

    # Handle outliers using IQR method
    Q1 = df_prophet["y"].quantile(0.25)
    Q3 = df_prophet["y"].quantile(0.75)
    IQR = Q3 - Q1
    df_prophet = df_prophet[
        (df_prophet["y"] >= Q1 - 1.5 * IQR) & (df_prophet["y"] <= Q3 + 1.5 * IQR)
    ]

    # Initialize and fit the Prophet model with adjusted parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        changepoint_range=0.8,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # Add the additional regressors
    model.add_regressor("Unit_Price")
    model.add_regressor("Lead_Time_Days")
    model.add_regressor("On_Time_Delivery_Rate")
    feature_names = ["Category", "Supplier_ID"]

    # Add encoded categorical features
    for feature in feature_names:
        model.add_regressor(feature)

    model.fit(df_prophet)

    # Convert the forecast duration into days based on the unit selected
    if duration_unit == "weeks":
        forecast_days = forecast_duration * 7
    elif duration_unit == "months":
        forecast_days = forecast_duration * 30
    else:
        forecast_days = forecast_duration

    # Make future predictions for the next forecast_days
    future = model.make_future_dataframe(periods=forecast_days)

    # Add the additional features to the future dataframe
    last_known_features = (
        df_prophet[
            ["ds", "Unit_Price", "Lead_Time_Days", "On_Time_Delivery_Rate"]
            + list(feature_names)
        ]
        .iloc[-1]
        .to_dict()
    )
    future_features = pd.DataFrame([last_known_features] * len(future))
    future_features["ds"] = future["ds"]
    future = pd.concat([future, future_features.drop("ds", axis=1)], axis=1)

    # Fill NaN values
    for column in future.columns:
        if column != "ds":
            future[column] = future[column].fillna(df_prophet[column].mean())

    forecast = model.predict(future)

    # Extract future predictions starting after the last date in the dataset
    last_date = df_prophet["ds"].max()
    future_forecast = forecast[forecast["ds"] > last_date]

    # Calculate the reorder point and safety stock
    forecast_table = future_forecast[["ds", "yhat"]].rename(
        columns={"ds": "Date", "yhat": "Forecasted Demand"}
    )
    forecast_table["Forecasted Demand"] = (
        forecast_table["Forecasted Demand"].round().astype(int)
    )

    mean_demand_per_day = forecast_table["Forecasted Demand"].mean()
    std_demand_per_day = forecast_table["Forecasted Demand"].std()

    mean_demand_lead_time = mean_demand_per_day * lead_time
    std_demand_lead_time = np.sqrt(
        (lead_time * std_demand_per_day**2) + (mean_demand_per_day**2 * 2**2)
    )

    z_score = norm.ppf(service_level)
    
    # Handle potential NaN values
    if np.isnan(std_demand_lead_time):
        safety_stock = 0
        reorder_point = int(mean_demand_lead_time)
    else:
        safety_stock = int(max(0, z_score * std_demand_lead_time))
        reorder_point = int(mean_demand_lead_time + safety_stock)

    # Perform cross-validation
    df_cv = cross_validation(
        model, initial="730 days", period="180 days", horizon="365 days"
    )
    df_p = performance_metrics(df_cv)

    # Ensure the forecast data covers the correct date range
    last_date = df_prophet["ds"].max()
    future_forecast = forecast[forecast["ds"] > last_date]
    future_forecast = future_forecast.iloc[:forecast_days]  # Limit to requested forecast duration

    st.session_state["forecast_data"] = future_forecast
    st.session_state["safety_stock"] = safety_stock
    st.session_state["reorder_point"] = reorder_point
    
    st.session_state["forecast_shown"] = True

    return future_forecast, safety_stock, reorder_point, df_p