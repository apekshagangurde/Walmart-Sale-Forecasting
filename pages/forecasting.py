import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Sample data generation
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 300, size=len(dates)),
    'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=len(dates)),
    'promotion': np.random.choice(['None', 'Discount', 'Buy One Get One'], size=len(dates))
})

data['month'] = data['date'].dt.to_period('M')
monthly_sales = data.groupby('month')['sales'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()

st.title("Sales Forecasting")
st.write("""
Use this page to select a month or year and forecast the sales for that period.
""")

# User input for forecasting
forecast_period = st.selectbox("Select forecast period:", ["Next Month", "Next Year"])
forecast_start_date = st.date_input("Start Date", value=datetime.now())

# Prepare data for forecasting
features = ['weather', 'promotion']
data_encoded = pd.get_dummies(data[features])
X = data_encoded
y = data['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Generate forecast data
if forecast_period == "Next Month":
    future_dates = pd.date_range(start=forecast_start_date, periods=30, freq='D')
elif forecast_period == "Next Year":
    future_dates = pd.date_range(start=forecast_start_date, periods=365, freq='D')

forecast_data = pd.DataFrame({'date': future_dates})
forecast_data['month'] = forecast_data['date'].dt.to_period('M')
forecast_data_encoded = pd.get_dummies(forecast_data[['month']])
forecast_data_encoded = forecast_data_encoded.reindex(columns=X.columns, fill_value=0)
forecast_data['forecast'] = model.predict(forecast_data_encoded)

# Plotting sales trends
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['month'], monthly_sales['sales'], color='blue', linestyle='-', marker='o', label='Historical Sales')
plt.plot(forecast_data['date'], forecast_data['forecast'], color='red', linestyle='--', marker='o', label='Forecast Sales')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trends and Forecast")
plt.legend()
st.pyplot(plt)

# Summary based on forecast
st.write(f"""
### Summary

- **Forecast Period:** {forecast_period}
- **Selected Start Date:** {forecast_start_date}
- **Predicted Sales:** {forecast_data['forecast'].sum():.2f}
""")
