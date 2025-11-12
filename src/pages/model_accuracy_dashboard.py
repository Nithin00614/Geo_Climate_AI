# src/pages/model_accuracy_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from evaluate_models import evaluate_all_cities

st.set_page_config(page_title="📊 Model Accuracy Dashboard", layout="wide")
st.title("📊 GeoClimate AI — Model Accuracy Evaluation")

metrics_path = "metrics/model_accuracy.csv"
if st.button("🔄 Evaluate All Models"):
    with st.spinner("Evaluating models..."):
        df = evaluate_all_cities()
else:
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
    else:
        df = pd.DataFrame()

if df is not None and not df.empty:
    st.success(f"✅ Evaluated {len(df)} city models.")
    st.dataframe(df, use_container_width=True)

    # Bar chart — RMSE
    st.subheader("📉 RMSE (Lower is Better)")
    fig_rmse = px.bar(df, x="city", y="RMSE", title="City-wise RMSE", color="RMSE")
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Bar chart — MAE
    st.subheader("📊 MAE (Lower is Better)")
    fig_mae = px.bar(df, x="city", y="MAE", title="City-wise MAE", color="MAE")
    st.plotly_chart(fig_mae, use_container_width=True)

    # Bar chart — R²
    st.subheader("📈 R² (Higher is Better)")
    fig_r2 = px.bar(df, x="city", y="R2", title="City-wise R² Scores", color="R2")
    st.plotly_chart(fig_r2, use_container_width=True)
else:
    st.warning("⚠️ No metrics found. Click 'Evaluate All Models' to generate metrics.")
