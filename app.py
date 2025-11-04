import streamlit as st
import pandas as pd
import os
from src.data_loader import fetch_weather_data
from src.preprocess import preprocess_data
from src.predict import predict_temperature
from src.visualize import plot_temperature_trends, plot_actual_vs_predicted

# ---------------------------#
# 🎨 Page Config
# ---------------------------#
st.set_page_config(
    page_title="GeoClimate-AI Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# Custom CSS for gradient header and layout
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
            padding: 20px;
        }
        .title {
            font-size: 38px !important;
            color: #0077b6;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            font-size: 18px;
            color: #444;
            text-align: center;
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #0077b6;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #0096c7;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------#
# 🌍 App Title
# ---------------------------#
st.markdown("<div class='title'>GeoClimate-AI Dashboard 🌦️</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Weather Insights & Climate Prediction</div>", unsafe_allow_html=True)
st.divider()

# ---------------------------#
# ⚙️ Sidebar
# ---------------------------#
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Fetch Latest Weather Data"):
    with st.spinner("Fetching live weather data..."):
        df = fetch_weather_data()
        st.success("✅ Weather data updated successfully!")
        st.dataframe(df)

# ---------------------------#
# 📊 Load or Preprocess Data
# ---------------------------#
data_path = os.path.join("data", "weather_data_clean.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.subheader("📂 Cleaned Weather Data (Recent Records)")
    st.dataframe(df.tail(10))

    # ---------------------------#
    # 📈 Summary Metrics
    # ---------------------------#
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Avg Temp (°C)", f"{df['temperature'].mean():.2f}")
    col2.metric("💧 Avg Humidity (%)", f"{df['humidity'].mean():.2f}")
    col3.metric("🌬️ Avg Wind Speed (m/s)", f"{df['wind_speed'].mean():.2f}")
    col4.metric("📊 Avg Pressure (hPa)", f"{df['pressure'].mean():.2f}")

else:
    st.warning("⚠️ No processed data found. Please run main.py once to create it.")
    df = None

st.divider()

# ---------------------------#
# 🤖 AI Prediction Section
# ---------------------------#
st.subheader("🤖 Temperature Prediction Engine")

col1, col2, col3 = st.columns(3)
humidity = col1.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
pressure = col2.number_input("Pressure (hPa)", min_value=800, max_value=1100, value=1012)
wind_speed = col3.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.5)

if st.button("🔍 Predict Temperature"):
    predicted_temp = predict_temperature(humidity, pressure, wind_speed)
    st.success(f"🌡️ Predicted Temperature: {predicted_temp:.2f} °C")

st.divider()

# ---------------------------#
# 📊 Visualizations
# ---------------------------#
if df is not None:
    st.subheader("📈 Weather Data Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_temperature_trends(df))
    with col2:
        st.pyplot(plot_actual_vs_predicted(df))

else:
    st.info("No data available to visualize yet.")
