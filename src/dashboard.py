import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(page_title="GeoClimate-AI Dashboard", layout="wide")
st.title("🌍 GeoClimate-AI Dashboard")
st.markdown("#### Advanced AI-based Climate Prediction & Visualization")

# ==========================
# LOAD DATA
# ==========================
DATA_PATH = "data/weather_data.csv"
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(DATA_PATH):
    st.error("❌ Data file not found! Please ensure 'data/weather_data.csv' exists.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Identify the correct date column automatically
if "date_time" in df.columns:
    df["date_time"] = pd.to_datetime(df["date_time"])
    date_col = "date_time"
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    date_col = "date"
else:
    st.error("⚠️ No date or date_time column found in the dataset.")
    st.stop()

# ==========================
# SIDEBAR FILTERS
# ==========================
st.sidebar.header("🔍 Filters")
cities = df["city"].unique().tolist()
city = st.sidebar.selectbox("Select a City", cities)

# Filter dataset
city_df = df[df["city"] == city]

# ==========================
# TEMPERATURE TREND VISUALIZATION
# ==========================
st.subheader(f"📈 Temperature Trends for {city}")

fig1 = px.line(
    city_df,
    x=date_col,
    y="temperature",
    title=f"Temperature Trend in {city}",
    markers=True,
    color_discrete_sequence=["#FF6347"]
)
st.plotly_chart(fig1, use_container_width=True)

# ==========================
# STATISTICAL INSIGHTS
# ==========================
st.subheader(f"📊 Statistical Overview for {city}")
col1, col2, col3 = st.columns(3)
col1.metric("Average Temperature (°C)", round(city_df["temperature"].mean(), 2))
col2.metric("Average Humidity (%)", round(city_df["humidity"].mean(), 2))
col3.metric("Average Wind Speed (m/s)", round(city_df["wind_speed"].mean(), 2))

# ==========================
# LSTM MODEL FORECAST
# ==========================
st.subheader(f"🤖 7-Day AI Forecast for {city}")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.warning("⚠️ LSTM model or scaler not found! Train your model before forecasting.")
else:
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Prepare data for forecasting
        features = ["temperature", "humidity", "pressure", "wind_speed"]
        city_data = city_df[features].values
        scaled_data = scaler.transform(city_data)

        last_7_days = scaled_data[-7:]
        predictions = []

        # Predict next 7 days using sliding window
        for _ in range(7):
            X_input = np.expand_dims(last_7_days[-7:], axis=0)
            pred_scaled = model.predict(X_input)
            pred_original = scaler.inverse_transform(
                np.concatenate([pred_scaled, last_7_days[-1, 1:].reshape(1, -1)], axis=1)
            )[0][0]
            predictions.append(pred_original)
            next_entry = np.concatenate([pred_scaled.flatten(), last_7_days[-1, 1:]])
            last_7_days = np.vstack([last_7_days, next_entry])[1:]

        # Create forecast dataframe
        future_dates = pd.date_range(
            start=pd.to_datetime(city_df[date_col].max()), periods=7, freq="D"
        )
        forecast_df = pd.DataFrame({"date": future_dates, "predicted_temperature": predictions})

        # Plot forecast
        fig2 = px.line(
            forecast_df,
            x="date",
            y="predicted_temperature",
            markers=True,
            title=f"7-Day Temperature Forecast for {city}",
            color_discrete_sequence=["#1E90FF"]
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.success("✅ Forecast generated successfully!")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ==========================
# WEATHER CONDITIONS SUMMARY
# ==========================
st.subheader(f"🌦️ Weather Condition Summary - {city}")
weather_counts = city_df["weather"].value_counts().reset_index()
weather_counts.columns = ["Weather Type", "Count"]
fig3 = px.bar(
    weather_counts,
    x="Weather Type",
    y="Count",
    color="Weather Type",
    title=f"Weather Condition Distribution for {city}",
    color_discrete_sequence=px.colors.qualitative.Bold,
)
st.plotly_chart(fig3, use_container_width=True)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("🚀 **GeoClimate-AI** — powered by Deep Learning & Climate Analytics 🌡️")
