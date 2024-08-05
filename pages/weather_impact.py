import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data generation
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 300, size=len(dates)),
    'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=len(dates)),
    'promotion': np.random.choice(['None', 'Discount', 'Buy One Get One'], size=len(dates)),
    'festival': np.random.choice(['None', 'Diwali', 'Christmas', 'Eid'], size=len(dates))
})

st.title("Weather Impact on Sales")
st.write("""
Explore how different weather conditions impact sales.
""")

# Weather impact analysis
weather_sales = data.groupby('weather')['sales'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='weather', y='sales', data=weather_sales, palette='viridis')
plt.title('Average Sales by Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Average Sales')
st.pyplot(plt)

# Summary
st.write("""
### Summary

- **Weather Impact:** Shows how different weather conditions affect average sales.
- **Insights:** For instance, Sunny days might generally lead to higher sales compared to Rainy or Cloudy days.
""")
