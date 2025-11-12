# src/pages/xai_dashboard.py
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from models.shap_explainer import explain_city_xgb

st.set_page_config(page_title="📘 Explainable AI Dashboard", layout="wide")
st.title("📘 GeoClimate AI — Explainable AI (SHAP Analysis)")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not files:
    st.warning("⚠️ No data found in /data/. Please train models first.")
    st.stop()

cities = [os.path.splitext(f)[0].capitalize() for f in files]
selected_city = st.selectbox("🏙️ Choose a city for SHAP analysis", cities)

if st.button("🔍 Generate Explainability Report"):
    st.info(f"Analyzing model explainability for **{selected_city}**...")
    file_path = os.path.join(DATA_DIR, f"{selected_city.lower()}.csv")

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("No data available for this city.")
        else:
            with st.spinner("Computing SHAP feature importance..."):
                fig, shap_values = explain_city_xgb(df, selected_city)
                st.success("✅ Explainability report generated successfully!")

                st.subheader(f"📊 Feature Importance for {selected_city}")
                st.pyplot(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🧠 What This Means")
                st.info("""
                • Features on the top have the greatest influence on predictions.  
                • Red = high feature value, Blue = low feature value.  
                • For example, if Humidity has high SHAP value, it strongly affects predicted temperature.  
                """)
    except Exception as e:
        st.error(f"❌ SHAP analysis failed: {e}")
