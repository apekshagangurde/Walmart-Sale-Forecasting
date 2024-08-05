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

st.title("Promotion Impact on Sales")
st.write("""
Analyze how different promotions influence the sales.
""")

# Promotion impact analysis
promotion_sales = data.groupby('promotion')['sales'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='promotion', y='sales', data=promotion_sales, palette='magma')
plt.title('Average Sales by Promotion Type')
plt.xlabel('Promotion')
plt.ylabel('Average Sales')
st.pyplot(plt)

# Summary
st.write("""
### Summary

- **Promotion Impact:** Shows how different promotion types affect average sales.
- **Insights:** For instance, promotions like "Buy One Get One" may lead to higher sales compared to "None".
""")
