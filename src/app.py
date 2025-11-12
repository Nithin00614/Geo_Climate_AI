# src/app.py
import streamlit as st

st.set_page_config(page_title="🌦️ GeoClimate AI", layout="wide")

st.title("🌍 GeoClimate AI — Integrated Predictive Climate Intelligence")

st.markdown("""
### Welcome to GeoClimate AI 🌎  
Select a dashboard from the sidebar to get started:
- **🌤️ AI Forecast & Model Dashboard** → 14-day predictions, model training, and visualization.  
- **⚠️ IoT & Climate Risk Dashboard** → live IoT sensor data, alerts, and risk visualization.  
""")

st.sidebar.success("Select a dashboard from the sidebar 👉")
