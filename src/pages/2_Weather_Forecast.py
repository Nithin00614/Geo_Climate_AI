import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from src.utils.data_loader import load_city_data
from src.models.predict import simple_forecast

st.title("⛅ Weather Forecast (14-Day)")

# Load all cities
try:
    cities = list(load_city_data.__globals__["load_all_city_data"]().keys())
except Exception:
    st.error("❌ Cannot load city list. Make sure CSVs exist in src/data/real_weather/")
    st.stop()

if not cities:
    st.error("❌ No weather dataset detected. Add CSV files to: `src/data/real_weather/`")
    st.stop()

city = st.selectbox("Select City", cities)

# Load selected city
df = load_city_data(city)

if df is None or len(df) < 60:
    st.error("⚠ Not enough data for forecasting. Need at least **60 rows**.")
    st.stop()

st.success(f"📦 Loaded **{len(df)} records** for **{city}**")

# ----------------------------
# FORECAST
# ----------------------------
latest_real = df.tail(30)

forecast_days = 14
forecast_df = simple_forecast(df["temperature"], forecast_days)

future_dates = pd.date_range(
    df["date"].max() + pd.Timedelta(days=1),
    periods=forecast_days
)

forecast_df["date"] = future_dates

# ----------------------------
# PLOT
# ----------------------------
fig = go.Figure()

# Actual last 30 days
fig.add_trace(go.Scatter(
    x=latest_real["date"],
    y=latest_real["temperature"],
    mode="lines+markers",
    name="Actual (Last 30 Days)",
    line=dict(color="cyan")
))

# Forecast
fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_df["forecast"],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="yellow")
))

# Confidence bands
fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_df["ci_lower"],
    mode="lines",
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_df["ci_upper"],
    fill='tonexty',
    fillcolor="rgba(255,255,0,0.15)",
    mode="lines",
    line=dict(width=0),
    name="95% Confidence Interval"
))

fig.update_layout(
    title=f"📈 14-Day Temperature Forecast — {city}",
    yaxis_title="Temperature (°C)",
    template="plotly_dark",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
