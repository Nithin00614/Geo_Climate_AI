# src/pages/forecast_dashboard.py
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import traceback

from lstm_model import train_lstm_model, auto_train_all_cities
from models.ensemble_forecast import ensemble_forecast
from weather_fetcher import fetch_weather_data

st.set_page_config(page_title="🌤️ GeoClimate AI Forecast Dashboard", layout="wide")
st.title("🌤️ GeoClimate AI — Forecast & Model Dashboard")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Function to load data from all CSVs in /data ---
def load_all_city_data():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    dfs = []
    for f in files:
        try:
            temp_df = pd.read_csv(os.path.join(DATA_DIR, f))
            temp_df.columns = [c.lower().strip() for c in temp_df.columns]
            if "temp" in temp_df.columns and "temperature" not in temp_df.columns:
                temp_df.rename(columns={"temp": "temperature"}, inplace=True)
            if "city" not in temp_df.columns:
                city_name = os.path.splitext(f)[0]
                temp_df["city"] = city_name
            dfs.append(temp_df)
        except Exception as e:
            print("Skipping:", f, "Error:", e)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# --- Load Dataset ---
df = load_all_city_data()

if df.empty:
    st.warning("⚠️ No data found — fetching sample city data...")
    df = fetch_weather_data("bengaluru")

for col in ["temperature", "humidity", "rainfall", "city", "date"]:
    if col not in df.columns:
        if col == "city":
            df["city"] = "unknown"
        elif col == "date":
            df["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
        elif col == "temperature":
            df["temperature"] = np.random.uniform(22, 30, len(df))
        else:
            df[col] = 0

df["city"] = df["city"].astype(str)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
cities = sorted(df["city"].unique())
st.success(f"✅ Loaded data for {len(cities)} cities: {', '.join(cities[:5])}...")

# --- City Selection ---
selected_city = st.selectbox("🏙️ Select a City", cities)

with st.expander("⚙️ Model Training & Data Options"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Train Model for Selected City"):
            m, s = train_lstm_model(df, selected_city)
            if m:
                st.success("✅ Model trained successfully.")
            else:
                st.error("❌ Model training failed.")
    with col2:
        if st.button("🤖 Auto-Train All Cities"):
            auto_train_all_cities(df)
            st.success("✅ All city models trained successfully.")
    with col3:
        if st.button("🌤️ Refresh Weather Data (API)"):
            try:
                fetch_weather_data(selected_city)
                st.success("✅ Weather data updated successfully.")
            except Exception as e:
                st.error(f"❌ Weather fetch failed: {e}")

st.markdown("---")
st.header("📈 14-Day Unified AI Forecast (LSTM + Prophet + XGBoost Fusion)")

# --- Forecast Generation ---
if selected_city:
    try:
        st.info(f"Generating 14-day AI-fused forecast for **{selected_city}** ...")

        city_df = df[df["city"].str.lower() == selected_city.lower()].copy()
        if city_df.empty:
            st.warning("⚠️ No records found for this city — fetching data...")
            city_df = fetch_weather_data(selected_city)

        results = ensemble_forecast(city_df, selected_city)

        # --- Validate results ---
        if results is None or results.empty:
            st.warning("⚠️ Forecast could not be generated — check data or model.")
        else:
            # ✅ FIX: Rename Prophet-style 'ds' → 'date'
            if "ds" in results.columns:
                results.rename(columns={"ds": "date"}, inplace=True)

            # --- Display model weights ---
            w_lstm = float(results["lstm_weight"].iloc[0]) * 100
            w_prophet = float(results["prophet_weight"].iloc[0]) * 100
            w_xgb = float(results["xgb_weight"].iloc[0]) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("🔹 LSTM Weight", f"{w_lstm:.1f}%")
            col2.metric("🔹 Prophet Weight", f"{w_prophet:.1f}%")
            col3.metric("🔹 XGBoost Weight", f"{w_xgb:.1f}%")

            # --- Unified Forecast Plot ---
            st.subheader(f"📊 14-Day Unified Forecast — {selected_city}")
            fig = px.line(
                results,
                x="date",
                y="final_pred",
                title=f"🌦️ Unified AI Forecast for {selected_city}",
                markers=True,
                line_shape="spline"
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(
                title_font_size=20,
                yaxis_title="Predicted Temperature (°C)",
                xaxis_title="Date",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Forecast Table ---
            st.subheader("📋 Forecast Summary")
            st.dataframe(
                results[["date", "final_pred"]].rename(columns={"final_pred": "Predicted Temperature (°C)"}),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")
        st.code(traceback.format_exc())
