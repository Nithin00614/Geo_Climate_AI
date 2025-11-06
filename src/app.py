import streamlit as st
import pandas as pd
import os
from lstm_model import train_lstm_model, predict_next_7_days
from data_loader import load_weather_data
from preprocess import preprocess_weather_data


st.set_page_config(page_title="GeoClimate AI", page_icon="🌍", layout="wide")

st.title("🌦️ GeoClimate AI — Advanced Weather Forecasting (LSTM Model)")

# Section: City Input
st.sidebar.header("🌆 Choose Location")
city = st.sidebar.text_input("Enter City Name", "Delhi")

# Fetch or load data
data_path = os.path.join("data", "weather_data.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.warning("Weather data not found locally. Fetching new data...")
    df = fetch_weather_data(city)
    if df is not None:
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)

if df is not None:
    st.write(f"### 📈 Showing weather data for **{city}**")
    st.dataframe(df.tail(10))

    # Model training button
    if st.button("🔁 Train / Retrain LSTM Model"):
        with st.spinner("Training LSTM model..."):
            model, scaler = train_lstm_model(df, city)
        if model is not None:
            st.success("✅ Model trained successfully and saved in /models!")
        else:
            st.error("⚠️ Training failed. Try using a city with more data points.")

    # Forecast button
    if st.button("🌤️ Predict Next 7 Days"):
        model_path = "models/lstm_temperature_model.keras"
        scaler_path = "models/scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning("⚠️ LSTM model or scaler not found! Train your model first.")
        else:
            with st.spinner("Generating 7-day forecast..."):
                preds = predict_next_7_days(df, city)
            if preds is not None:
                st.success("✅ 7-Day Forecast Ready!")
                forecast_df = pd.DataFrame({
                    "Date": pd.date_range(datetime.now(), periods=7, freq="D"),
                    "Predicted Temperature (°C)": preds
                })
                st.line_chart(forecast_df.set_index("Date"))
                st.dataframe(forecast_df)
else:
    st.error("❌ Unable to load data. Please verify your city or API setup.")
