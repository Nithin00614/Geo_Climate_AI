# src/pages/iot_risk_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import random

st.set_page_config(layout="wide")

st.markdown("""
# 🚨 Climate Risk Alerts  
View recent alerts triggered by IoT sensors with CRI thresholds.
""")

st.title("⚠️ GeoClimate AI — IoT & Climate Risk Analytics")

st.markdown("""
### 🌍 Real-Time Climate Risk & IoT Dashboard  
This module combines **simulated IoT sensor readings** with **climate risk analysis**.  
It dynamically computes a **Climate Risk Index (CRI)** per city and visualizes risk levels.
""")

# --- Config ---
CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Kolkata", "Ahmedabad", "Pune"]
st.sidebar.header("⚙️ Simulation Settings")
interval = st.sidebar.slider("⏱️ Update Interval (sec)", 1, 10, 3)
updates = st.sidebar.number_input("🔁 Total Updates", 5, 100, 20)

# --- Helper ---
def compute_cri(temp, humidity, rainfall):
    t_score = np.clip((temp - 25) / 10, 0, 1)
    h_score = np.clip((humidity - 60) / 40, 0, 1)
    r_score = np.clip(rainfall / 50, 0, 1)
    cri = 100 * (0.5 * t_score + 0.3 * h_score + 0.2 * r_score)
    return round(cri, 2)

# --- Live Simulation ---
if st.button("▶️ Start IoT Simulation"):
    st.subheader("🌡️ Real-Time IoT Monitoring & Risk Index")
    container = st.empty()
    log_data = []

    for step in range(updates):
        frame = []
        for city in CITIES:
            temp = np.random.uniform(20, 38)
            hum = np.random.uniform(30, 90)
            rain = np.random.uniform(0, 40)
            cri = compute_cri(temp, hum, rain)
            frame.append({"city": city, "temperature": temp, "humidity": hum, "rainfall": rain, "CRI": cri})
            log_data.append({"iteration": step + 1, "city": city, "CRI": cri})

        df = pd.DataFrame(frame)
        mean_risk = df["CRI"].mean()

        with container.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌡️ Avg Temperature", f"{df['temperature'].mean():.2f}°C")
                st.metric("💧 Avg Humidity", f"{df['humidity'].mean():.2f}%")
                st.metric("🌧️ Avg Rainfall", f"{df['rainfall'].mean():.2f} mm")
                st.metric("⚠️ Mean Risk Index", f"{mean_risk:.2f}")
            with col2:
                fig = px.bar(df, x="city", y="CRI", color="CRI",
                             color_continuous_scale="RdYlGn_r",
                             title="🌆 City-wise Climate Risk Index (CRI)")
                st.plotly_chart(fig, use_container_width=True)

        time.sleep(interval)

    # --- Risk Map ---
    st.subheader("🗺️ Climate Risk Heatmap (Simulated Coordinates)")
    city_locs = {
        "Bengaluru": [12.97, 77.59],
        "Chennai": [13.08, 80.27],
        "Delhi": [28.61, 77.20],
        "Hyderabad": [17.38, 78.48],
        "Mumbai": [19.07, 72.87],
        "Kolkata": [22.57, 88.36],
        "Ahmedabad": [23.03, 72.58],
        "Pune": [18.52, 73.85]
    }
    map_df = pd.DataFrame([
        {"city": c, "lat": city_locs[c][0], "lon": city_locs[c][1], "CRI": random.uniform(20, 90)}
        for c in city_locs
    ])
    fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="CRI",
                                color_continuous_scale="YlOrRd", size="CRI",
                                zoom=4.2, height=550,
                                mapbox_style="carto-darkmatter",
                                title="🔥 Live Climate Risk Intensity Map (CRI)")
    st.plotly_chart(fig_map, use_container_width=True)

    st.success("✅ Simulation Completed.")
