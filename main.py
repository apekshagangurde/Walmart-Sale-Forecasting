import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose as season
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Set page configuration
st.set_page_config(page_title="Walmart Sales Analysis", layout="wide")

# Title of the app
st.title("Walmart Sales Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    df_store = pd.read_csv('data/stores.csv')
    df_train = pd.read_csv('data/train.csv')
    df_features = pd.read_csv('data/features.csv')
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    df.drop(['IsHoliday_y'], axis=1, inplace=True)
    df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
    df = df.loc[df['Weekly_Sales'] > 0]
    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    # Adding holiday columns
    df['Super_Bowl'] = df['Date'].isin(['2010-02-12', '2011-02-11', '2012-02-10'])
    df['Labor_Day'] = df['Date'].isin(['2010-09-10', '2011-09-09', '2012-09-07'])
    df['Thanksgiving'] = df['Date'].isin(['2010-11-26', '2011-11-25'])
    df['Christmas'] = df['Date'].isin(['2010-12-31', '2011-12-30'])
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
store = st.sidebar.selectbox("Select Store", df['Store'].unique())
dept = st.sidebar.selectbox("Select Department", df['Dept'].unique())
year = st.sidebar.slider("Select Year", int(df['year'].min()), int(df['year'].max()), (2010, 2012))

filtered_df = df[(df['Store'] == store) & (df['Dept'] == dept) & (df['year'] >= year[0]) & (df['year'] <= year[1])]

# Display data
st.write("## Filtered Data")
st.dataframe(filtered_df)

# Plotting
st.write("## Weekly Sales Analysis")

# Line plot of weekly sales
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=filtered_df, x='Date', y='Weekly_Sales', ax=ax)
ax.set_title("Weekly Sales Over Time")
st.pyplot(fig)

# Bar plot for average weekly sales by holiday
st.write("## Average Weekly Sales by Holiday")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=filtered_df, ax=ax)
ax.set_title("Average Weekly Sales by Holiday")
st.pyplot(fig)

# Super Bowl analysis
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=filtered_df, ax=ax)
ax.set_title("Super Bowl vs Weekly Sales")
st.pyplot(fig)

# Labor Day analysis
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Labor_Day', y='Weekly_Sales', data=filtered_df, ax=ax)
ax.set_title("Labor Day vs Weekly Sales")
st.pyplot(fig)

# Thanksgiving analysis
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=filtered_df, ax=ax)
ax.set_title("Thanksgiving vs Weekly Sales")
st.pyplot(fig)

# Christmas analysis
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Christmas', y='Weekly_Sales', data=filtered_df, ax=ax)
ax.set_title("Christmas vs Weekly Sales")
st.pyplot(fig)

# Plotting avg weekly sales according to holidays by types
st.write("## Average Weekly Sales by Holiday and Store Type")
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56, 10179.27, 9893.45, 8031.52]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x, B_means, width, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30, color='r')  # holidays avg
plt.axhline(y=15952.82, color='green')  # not-holiday avg

fig.tight_layout()

st.pyplot(fig)

# Seasonal decomposition
st.write("## Seasonal Decomposition")
seasonal_decompose_result = season(filtered_df.set_index('Date')['Weekly_Sales'], model='multiplicative')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
seasonal_decompose_result.observed.plot(ax=ax1, title='Observed')
seasonal_decompose_result.trend.plot(ax=ax2, title='Trend')
seasonal_decompose_result.seasonal.plot(ax=ax3, title='Seasonal')
seasonal_decompose_result.resid.plot(ax=ax4, title='Residual')
st.pyplot(fig)

# ARIMA model predictions
st.write("## ARIMA Model Predictions")
model = auto_arima(filtered_df.set_index('Date')['Weekly_Sales'], seasonal=True, m=52)
forecast = model.predict(n_periods=52)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_df.set_index('Date')['Weekly_Sales'].index[-52:], forecast, label="ARIMA Forecast")
ax.plot(filtered_df.set_index('Date')['Weekly_Sales'].index[-52:], filtered_df.set_index('Date')['Weekly_Sales'].iloc[-52:], label="Actual Sales")
ax.legend()
ax.set_title("ARIMA Model Predictions")
st.pyplot(fig)

# Show data summary
st.write("## Data Summary")
st.write(filtered_df.describe())

# Correlation heatmap
st.write("## Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(filtered_df.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
