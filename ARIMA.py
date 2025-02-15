import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error

# Load dataset
def load_data():
    date_rng = pd.date_range(start='1/1/2020', end='1/01/2023', freq='D')
    data = pd.DataFrame(date_rng, columns=['date'])
    data['value'] = np.sin(np.linspace(0, 50, len(date_rng))) + np.random.normal(scale=0.5, size=len(date_rng))
    data.set_index('date', inplace=True)
    return data

data = load_data()

# Split into train and test
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Streamlit UI
st.title("Time Series Forecasting")
model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA", "Prophet"])
forecast_period = st.slider("Select forecast period", min_value=5, max_value=60, value=30)

if model_choice == "ARIMA":
    p = st.number_input("p (Auto-Regressive Order)", min_value=0, max_value=5, value=2)
    d = st.number_input("d (Differencing Order)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (Moving Average Order)", min_value=0, max_value=5, value=2)
    model = ARIMA(train, order=(p, d, q)).fit()
    forecast = model.forecast(steps=forecast_period)

elif model_choice == "SARIMA":
    p = st.number_input("p (Auto-Regressive Order)", min_value=0, max_value=5, value=2)
    d = st.number_input("d (Differencing Order)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (Moving Average Order)", min_value=0, max_value=5, value=2)
    P = st.number_input("P (Seasonal Auto-Regressive Order)", min_value=0, max_value=5, value=1)
    D = st.number_input("D (Seasonal Differencing Order)", min_value=0, max_value=2, value=1)
    Q = st.number_input("Q (Seasonal Moving Average Order)", min_value=0, max_value=5, value=1)
    s = st.number_input("Seasonal Period", min_value=2, max_value=30, value=7)
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()
    forecast = model.forecast(steps=forecast_period)

elif model_choice == "Prophet":
    train_prophet = train.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(train_prophet)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast_df = model.predict(future)
    forecast = forecast_df.set_index('ds')['yhat'][-forecast_period:]

# Calculate RMSE
actuals = test[:forecast_period]
rmse = np.sqrt(mean_squared_error(actuals, forecast[:len(actuals)]))

# Plot results
st.write(f"RMSE: {rmse:.4f}")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=train, label='Train')
sns.lineplot(data=test, label='Test')
sns.lineplot(data=forecast, label='Forecast', linestyle='dashed')
st.pyplot(plt)
