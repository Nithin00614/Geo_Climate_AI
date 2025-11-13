# src/pages/iot_alert_dashboard.py
"""
🌍 IoT & Climate Risk Alert Dashboard

- Reads live IoT data and alerts from SQLite
- Auto-refreshes every few seconds
- Displays risk heatmap and recent alerts
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import time
import os
import streamlit as st
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(layout="wide")

st.markdown("""
# 📡 IoT Sensor Dashboard  
Live climate sensor data with CRI computation and geolocation tracking.
""")

DB_PATH = "data/climate_ai.db"

st.title("📡 Real-Time IoT & Climate Risk Alerts Dashboard")

REFRESH_INTERVAL = st.sidebar.slider("🔁 Auto-refresh interval (seconds)", 3, 60, 5)

# --- Load data functions ---
def load_iot_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM iot_data ORDER BY timestamp DESC LIMIT 300", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def load_alerts():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

# --- Main refresh loop ---
placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("📊 IoT Sensor Readings (Last 300)")
        iot_df = load_iot_data()
        alerts_df = load_alerts()

        if iot_df.empty:
            st.warning("⚠️ No IoT data found. Please start the simulator.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.scatter_mapbox(
                iot_df,
                lat="lat",
                lon="lon",
                color="cri",
                size="cri",
                hover_name="city",
                hover_data=["temperature", "humidity", "rainfall", "aqi"],
                color_continuous_scale="RdYlGn_r",
                zoom=4,
                height=500,
                title="🌡️ Live Climate Risk Map (CRI Levels)"
            )
            fig.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":30,"l":0,"b":0})

            # 👇 Add a unique key based on time to avoid duplicate IDs
            st.plotly_chart(fig, use_container_width=True, key=f"map_{time.time()}")


            with col2:
                st.metric("🌍 Total Cities", len(iot_df["city"].unique()))
                st.metric("📈 Avg CRI", round(iot_df["cri"].mean(), 2))
                st.metric("🔥 Max CRI", round(iot_df["cri"].max(), 2))

        st.divider()
        st.subheader("🚨 Recent Alerts (Top 50)")

        if alerts_df.empty:
            st.info("✅ No recent alerts. System is stable.")
        else:
            for _, row in alerts_df.iterrows():
                alert_color = "red" if row["level"] == "HIGH" else "orange"
                unique_key = f"alert_{row['id']}_{time.time()}"  # <— unique each cycle
                st.toast(f"🚨 {row['city']} | CRI: {row['cri']} ({row['level']}) — {row['message']}")
                st.markdown(
                    f"<div style='border-left:5px solid {alert_color}; padding:8px; margin:6px 0; "
                    f"border-radius:6px; background-color:rgba(255,255,255,0.05);'>"
                    f"<b>{row['timestamp']}</b> — <b>{row['city']}</b><br>"
                    f"CRI: <b>{row['cri']}</b> | Level: <b>{row['level']}</b><br>"
                    f"{row['message']}</div>",
                    unsafe_allow_html=True,
                    key=10,   # 👈 add key here
    )



    # Wait for refresh
    time.sleep(REFRESH_INTERVAL)
