import streamlit as st
import pandas as pd
import os
from lstm_model import train_lstm_model, predict_next_7_days, auto_train_all_cities
from weather_fetcher import fetch_weather_data
import joblib
import plotly.express as px

st.set_page_config(page_title="🌦️ GeoClimate AI Dashboard", layout="wide")
st.title("🌍 GeoClimate AI — Smart Climate Forecasting")

DATA_PATH = "data/weather_data.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Load or fetch data ---
if not os.path.exists(DATA_PATH):
    st.warning("⚠️ No data found — fetching sample city weather data...")
    df = fetch_weather_data("bengaluru")
else:
    df = pd.read_csv(DATA_PATH)

if df is not None:
    st.success(f"✅ Loaded data for {len(df['city'].unique())} cities.")
else:
    st.error("❌ Could not load any data.")
    st.stop()

cities = sorted(df["city"].unique())
selected_city = st.selectbox("🏙️ Choose a city", cities)

col1, col2, col3 = st.columns(3)

# --- Train for selected city ---
with col1:
    if st.button("🚀 Train Model for Selected City"):
        model, scaler = train_lstm_model(df, selected_city)
        if model:
            st.success(f"✅ Model trained for {selected_city}!")
        else:
            st.error(f"❌ Could not train model for {selected_city}.")

# --- Auto-train for all cities ---
with col2:
    if st.button("🤖 Auto-Train All Cities"):
        auto_train_all_cities(df)
        st.success("✅ All city models trained successfully!")

# --- Fetch real-time updates ---
with col3:
    if st.button("🌤️ Refresh Weather Data"):
        fetch_weather_data(selected_city)
        st.success("✅ Real-time data refreshed!")

# --- Forecast Section ---
st.subheader(f"📈 7-Day AI Forecast for {selected_city}")
model_path = f"models/lstm_{selected_city.lower()}.keras"
scaler_path = f"models/scaler_{selected_city.lower()}.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    forecast = predict_next_7_days(df, selected_city, scaler)

    if forecast is not None:
        fig = px.line(forecast, x="date", y="predicted_temperature",
                      title=f"🌡️ 7-Day Predicted Temperature for {selected_city}",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Forecast could not be generated.")
else:
    st.warning("⚠️ LSTM model or scaler not found! Please train your model before forecasting.")
