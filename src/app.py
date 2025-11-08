# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import time
from datetime import timedelta
from keras.models import load_model

from weather_fetcher import fetch_weather_data
from lstm_model import train_multifeature_lstm, auto_train_all_cities, predict_next_7_days
from iot_simulator import generate_mock_iot_data, batch_generate, CITY_COORDS
from alert_logger import log_alert, load_alert_history
from database_manager import init_db, save_forecast, save_iot_batch, save_alert
init_db()
print(f"🗄️ Using database at: {os.path.abspath('data/climate_ai.db')}")


st.set_page_config(page_title="🌦️ GeoClimate AI Dashboard", layout="wide")
st.title("🌍 GeoClimate AI — Smart Climate Forecasting, IoT & Alerts")

DATA_PATH = "data/historical_combined.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------
# Load dataset (or fetch sample)
# ------------------------
if not os.path.exists(DATA_PATH):
    st.warning("⚠️ No dataset found — fetching sample city data (Bengaluru)...")
    # fetch_weather_data should save CSV in data/ or return DataFrame
    df = fetch_weather_data("Bengaluru")
    if isinstance(df, pd.DataFrame):
        df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

if df is None or df.empty:
    st.error("❌ Could not load any data. Please run the data pipeline first.")
    st.stop()

# ensure date col
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

cities = sorted(df["city"].unique())
selected_city = st.selectbox("🏙️ Choose a city", cities)

# ----------------------------
# Training options and actions
# ----------------------------
col1, col2, col3 = st.columns(3)
with st.expander("⚙️ Training Options"):
    force_retrain = st.checkbox("🔁 Force retrain even if model already exists", value=False)
    st.info("If checked, all models will be retrained from scratch. Unchecked = skip existing models.")

with col1:
    if st.button("🚀 Train Model for Selected City"):
        with st.spinner(f"Training LSTM model for {selected_city}..."):
            train_multifeature_lstm(selected_city, force_retrain=force_retrain)
        st.success(f"✅ Training (done) for {selected_city}")

with col2:
    if st.button("🤖 Auto-Train All Cities"):
        with st.spinner("Auto-training models..."):
            if force_retrain:
                for c in df["city"].unique():
                    train_multifeature_lstm(c, force_retrain=True)
            else:
                auto_train_all_cities(df)
        st.success("✅ Auto-training complete")

with col3:
    if st.button("🌤️ Refresh Weather Data (API)"):
        with st.spinner("Fetching latest weather data..."):
            fetch_weather_data(selected_city)
            # reload dataset if your fetch writes to DATA_PATH
            if os.path.exists(DATA_PATH):
                df = pd.read_csv(DATA_PATH)
        st.success("✅ Weather data refreshed")

# ----------------------------
# Forecast Section
# ----------------------------
st.subheader(f"📈 7-Day AI Forecast for {selected_city}")

try:
    forecast = predict_next_7_days(df, selected_city)
except Exception as e:
    forecast = None
    st.warning(f"⚠️ Forecast generation failed: {e}")

if forecast is not None:
    # ✅ Save forecast results to DB
    try:
        save_forecast(selected_city, forecast, model_name="LSTM")
    except Exception as e:
        st.warning(f"⚠️ Failed to save forecast to DB: {e}")

    # ✅ Plot forecast
    fig = px.line(
        forecast,
        x="date",
        y="predicted_temperature",
        title=f"🌡️ 7-Day Predicted Temperature for {selected_city}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Risk levels in table
    forecast["Risk Level"] = "Low"
    forecast.loc[forecast["predicted_temperature"] > 38, "Risk Level"] = "High"
    forecast.loc[
        (forecast["predicted_temperature"] > 32) & (forecast["predicted_temperature"] <= 38),
        "Risk Level"
    ] = "Moderate"

    st.dataframe(
        forecast.reset_index(drop=True).style.background_gradient(subset=["predicted_temperature"]),
        use_container_width=True
    )
else:
    st.warning("⚠️ Forecast not available — train the model or check data.")


# ----------------------------
# Live IoT Mode
# ----------------------------
st.markdown("---")
st.subheader("📡 Live IoT Simulation & Alerts")

# session state for live mode
if "iot_running" not in st.session_state:
    st.session_state.iot_running = False

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.session_state.iot_running:
        if st.button("⏹ Stop Live IoT"):
            st.session_state.iot_running = False
    else:
        if st.button("▶ Start Live IoT"):
            st.session_state.iot_running = True

    freq = st.number_input("Update interval (seconds)", min_value=1, max_value=30, value=3, step=1)
    samples = st.number_input("Total updates (press Start to run this many)", min_value=1, max_value=1000, value=50, step=1)

with col_b:
    st.write("Live IoT shows simulated sensor readings and compares against AI forecast.")

placeholder = st.empty()
alerts_placeholder = st.empty()

# helper to compute a short forecast summary (mean predicted temp)
def get_forecast_mean(city):
    f = predict_next_7_days(df, city)
    if f is None:
        return None
    return float(f["predicted_temperature"].mean())

# run live updates loop (will run until Stop button pressed)
if st.session_state.iot_running:
    # precompute city list for batch; use only trained cities + selected city
    city_list = list(cities)
    # If too many cities, limit to first 20 to avoid heavy load
    city_list = city_list[:20]

    # compute AI forecast baseline mean temps for each city if model exists
    baseline = {}
    for c in city_list:
        try:
            baseline[c] = get_forecast_mean(c)
        except Exception:
            baseline[c] = None

    i = 0
    while st.session_state.iot_running and i < samples:
        # generate a batch of readings for all cities
        readings = batch_generate(city_list)
        live_df = pd.DataFrame(readings)

        # save to DB for persistence
        try:
            save_iot_batch(live_df)
        except Exception as e:
            st.warning(f"⚠️ Failed to save IoT data: {e}")

        live_df["timestamp"] = pd.to_datetime(live_df["timestamp"])

        # iterate and show selected city prominently
        selected_row = live_df[live_df["city"] == selected_city].iloc[0]
        sel_temp = selected_row["temperature"]
        sel_hum = selected_row["humidity"]
        sel_rain = selected_row["rainfall"]

        with placeholder.container():
            st.markdown(f"**Update #{i+1} — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}**")
            cols = st.columns(4)
            cols[0].metric(label=f"🌡️ {selected_city} Temperature (°C)", value=f"{sel_temp}")
            cols[1].metric(label=f"💧 Humidity (%)", value=f"{sel_hum}")
            cols[2].metric(label=f"🌧️ Rainfall (mm)", value=f"{sel_rain}")
            # show simple comparison to AI prediction (if available)
            baseline_temp = baseline.get(selected_city)
            if baseline_temp is not None:
                diff = sel_temp - baseline_temp
                cols[3].metric(label="Δ vs AI mean (°C)", value=f"{diff:.2f}")

            st.write("### Top 10 live readings (recent)")
            st.dataframe(live_df.sort_values("timestamp", ascending=False).head(10).reset_index(drop=True))

        # Alerts logic
        alert_msgs = []
        # heatwave: IoT temp significantly > predicted mean
        if baseline.get(selected_city) is not None and sel_temp > baseline[selected_city] + 2:
            alert_msgs.append(f"🔥 Heatwave alert in {selected_city}: sensor {sel_temp}°C exceeds AI mean by {sel_temp - baseline[selected_city]:.2f}°C")
        # heavy rainfall
        if sel_rain > 5:
            alert_msgs.append(f"🌧️ Heavy rainfall alert in {selected_city}: {sel_rain} mm")
        # extremely low humidity
        if sel_hum < 40:
            alert_msgs.append(f"⚠️ Very low humidity in {selected_city}: {sel_hum}%")

        # Display alerts
        from alert_logger import log_alert, load_alert_history

        # Display alerts + log them persistently
        with alerts_placeholder.container():
            if alert_msgs:
                for m in alert_msgs:
                    st.error(m)
                    # Log alert type automatically
                    if "Heatwave" in m:
                        log_alert(selected_city, "Heatwave", m)
                        save_alert(selected_city, "Heatwave", m)
                    elif "rainfall" in m:
                        log_alert(selected_city, "Rainfall", m)
                        save_alert(selected_city, "Rainfall", m)
                    elif "humidity" in m:
                        log_alert(selected_city, "LowHumidity", m)
                        save_alert(selected_city, "LowHumidity", m)
                    else:
                        log_alert(selected_city, "General", m)
                        save_alert(selected_city, "General", m)
            else:
                st.success("✅ No critical alerts for selected city")


        i += 1
        time.sleep(freq)
        # force rerun check (if user toggled stop in another interaction)
        if not st.session_state.iot_running:
            break

    # stop at end
    st.session_state.iot_running = False
    st.success("Live IoT session finished.")

# ----------------------------
# Map: show cities colored by basic risk (based on AI mean)
# ----------------------------
st.markdown("---")
st.subheader("🗺️ City Risk Map (AI mean temperature)")

# prepare map data for cities we have coords for
map_rows = []
for c in cities:
    mean_temp = None
    try:
        mean_temp = get_forecast_mean(c)
    except Exception:
        mean_temp = None
    lat, lon = CITY_COORDS.get(c, (None, None))
    if lat is None or lon is None:
        continue
    risk = "Unknown"
    if mean_temp is None:
        risk = "No Model"
    elif mean_temp > 38:
        risk = "High"
    elif mean_temp > 32:
        risk = "Moderate"
    else:
        risk = "Low"
    map_rows.append({"city": c, "lat": lat, "lon": lon, "mean_temp": mean_temp, "risk": risk})

if map_rows:
    map_df = pd.DataFrame(map_rows)
    # color mapping
    color_map = {"High": "red", "Moderate": "orange", "Low": "green", "No Model": "gray", "Unknown": "gray"}
    map_df["color"] = map_df["risk"].map(color_map)

    fig_map = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        hover_name="city",
        size_max=12,
        color="risk",
        projection="natural earth",
        title="City Risk Overview (based on AI forecast mean temperature)"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No city coordinates available for map display.")

# ----------------------------
# Footer: quick controls / tips
# ----------------------------
st.markdown("---")
st.caption("Tip: Use 'Force retrain' after updating the data pipeline to rebuild models with latest data.")



# ----------------------------
# 📜 View Alert History
# ----------------------------
st.markdown("---")
st.subheader("📜 View Alert History")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Refresh Alert History"):
        st.session_state["alerts_df"] = load_alert_history()
    export_btn = st.download_button(
        "💾 Export as CSV",
        data=load_alert_history().to_csv(index=False).encode("utf-8"),
        file_name="alert_history.csv",
        mime="text/csv"
    )

with col2:
    if "alerts_df" not in st.session_state:
        st.session_state["alerts_df"] = load_alert_history()
    alerts_df = st.session_state["alerts_df"]

    if alerts_df.empty:
        st.info("No alerts recorded yet. Run IoT simulation to generate alerts.")
    else:
        st.dataframe(
            alerts_df.sort_values("Timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True
        )



# ----------------------------
# 📊 Alert Analytics & Risk Intelligence Dashboard
# ----------------------------
from alert_analytics import (
    load_alerts_df,
    compute_city_alert_stats,
    compute_risk_score,
    plot_alert_trends,
    plot_top_cities,
    plot_alert_type_distribution
)

st.markdown("---")
st.subheader("📊 Alert Analytics & City Risk Intelligence")

# Load alerts
alerts_df = load_alerts_df()

if alerts_df.empty:
    st.info("No alerts logged yet. Run IoT simulation to generate alert data.")
else:
    # compute analytics
    alert_stats = compute_city_alert_stats(alerts_df)
    risk_df = compute_risk_score(alert_stats, df)

    # layout
    col1, col2 = st.columns(2)
    with col1:
        fig_trend = plot_alert_trends(alerts_df)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        fig_pie = plot_alert_type_distribution(alerts_df)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_top = plot_top_cities(alert_stats)
        if fig_top:
            st.plotly_chart(fig_top, use_container_width=True)
    with col4:
        st.dataframe(
            risk_df.sort_values("RiskScore", ascending=False).reset_index(drop=True),
            use_container_width=True
        )

    # map visualization
    st.markdown("### 🗺️ City Risk Map (Combined Risk Index)")
    if not risk_df.empty:
        color_scale = ["green", "yellow", "orange", "red"]
        fig_risk_map = px.scatter_geo(
            risk_df,
            lat="lat",
            lon="lon",
            hover_name="City",
            color="RiskScore",
            size="TotalAlerts",
            color_continuous_scale=color_scale,
            title="🌍 City Risk Level (AI + Alerts)"
        )
        st.plotly_chart(fig_risk_map, use_container_width=True)
