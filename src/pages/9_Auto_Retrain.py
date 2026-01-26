# src/pages/auto_retrain_dashboard.py
import streamlit as st
import pandas as pd
import os
from auto_retrain import auto_retrain_models
from models.lstm_model import train_lstm_model
from models.xgb_model import train_xgb_model
from models.ensemble_forecast import ensemble_forecast

from utils.banner import top_banner
top_banner()
    
st.set_page_config(layout="wide")

st.markdown("""
# 🔄 Auto Retrain  
Schedule or trigger automated model retraining on fresh data.
""")
st.title("🤖 GeoClimate AI — Automated Model Retraining")

metrics_path = "metrics/model_accuracy.csv"
retrain_log = "metrics/retrain_log.csv"

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Evaluate & Auto-Retrain Models"):
        with st.spinner("Evaluating performance and retraining underperformers..."):
            result = auto_retrain_models()
        st.success("✅ Auto-retraining complete. Metrics updated.")

with col2:
    if st.button("📊 View Latest Metrics"):
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ No metrics file found. Run evaluation first.")

st.markdown("---")
st.subheader("🧾 Retraining Log History")
if os.path.exists(retrain_log):
    log_df = pd.read_csv(retrain_log)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("ℹ️ No retraining log yet. Run auto-retrain first.")
