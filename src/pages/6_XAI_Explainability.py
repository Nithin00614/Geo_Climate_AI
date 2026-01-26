import streamlit as st
import pandas as pd, os, sys
from pathlib import Path

from src.models.xgb_model import load_xgb_model
from src.models.lstm_model import load_lstm_model
from src.utils.sidebar_style import apply_sidebar_style
from src.utils.header import page_header
from src.utils.banner import top_banner

apply_sidebar_style()
st.set_page_config(page_title="🧠 XAI Explainability", layout="wide")
top_banner()
page_header("XAI Explainability", "SHAP & Feature importance for model insights", "🧠")

DATA_DIR = Path("data")
MODELS_DIR = Path("src/models")

st.markdown("This page shows SHAP explanations for XGBoost (if available). If SHAP is not installed or model missing, a fallback summary is shown.")

# Try to import shap safely
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Load cities from data folder
try:
    from src.utils.data_loader import load_all_city_data
    all_data = load_all_city_data()
    cities = list(all_data.keys())
except Exception as e:
    st.error(f"Error loading cities: {str(e)}")
    cities = []

if not cities:
    st.warning("No city datasets found.")
    st.stop()

city = st.selectbox("Select City", sorted(cities))

if city not in all_data:
    st.error(f"Could not load data for {city}")
    st.stop()

df = all_data[city]

# Try load xgb model
try:
    model = load_xgb_model(city, save_dir="src/models/xgb")
except Exception as e:
    model = None
    st.warning(f"Could not load model: {str(e)}")

if model is None:
    st.warning("⚠️ XGBoost model not found for this city. Train models first in Model Manager.")
    st.info("SHAP explainability requires a trained XGBoost model.")
    st.stop()

st.success("XGB model loaded. Generating SHAP explanations...")

# prepare features
feat_cols = ['temperature','humidity','wind_speed','rainfall']
for c in feat_cols:
    if c not in df.columns:
        df[c] = 0.0

X = df[feat_cols].tail(200)  # sample recent rows

if not SHAP_AVAILABLE:
    st.warning("SHAP not installed. Install shap to see detailed explanations. Showing simple feature importances.")
    try:
        import numpy as np
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            fmap = pd.DataFrame({"feature":feat_cols, "importance":importances})
            st.bar_chart(fmap.set_index('feature'))
        else:
            st.info("Model does not expose feature_importances_.")
    except Exception as e:
        st.error(f"Could not compute importances: {e}")
    st.stop()

# SHAP path
explainer = shap.TreeExplainer(model) if hasattr(shap, "TreeExplainer") else None
if explainer is None:
    st.error("SHAP explainer not available for this model type.")
    st.stop()

with st.spinner("Computing SHAP values (may take a few seconds)..."):
    shap_values = explainer.shap_values(X)
    st.subheader("SHAP Summary (bar)")
    st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar", show=False))
    st.subheader("SHAP Summary (dot)")
    st.pyplot(shap.summary_plot(shap_values, X, show=False))

st.caption("XAI explanations help interpret model predictions and explain high-risk days.")
