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

st.title("Festivals Impact on Sales")
st.write("""
Explore how different festivals impact sales.
""")

# Festival impact analysis
festival_sales = data.groupby('festival')['sales'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='festival', y='sales', data=festival_sales, palette='coolwarm')
plt.title('Average Sales by Festival')
plt.xlabel('Festival')
plt.ylabel('Average Sales')
st.pyplot(plt)

# Summary
st.write("""
### Summary

- **Festival Impact:** Shows how different festivals affect average sales.
- **Insights:** For instance, festivals like Diwali and Christmas may lead to higher sales compared to non-festival periods.
""")
