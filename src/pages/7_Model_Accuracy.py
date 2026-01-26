import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.models.forecast_engine import ensemble_forecast
from src.utils.data_loader import load_all_city_data


st.markdown("# 📏 Model Accuracy — Diagnostics")
st.caption("Backtested performance comparison over last 14 available days.")


# --------------------------
# USER INPUTS
# --------------------------
all_cities = load_all_city_data()
cities = sorted(all_cities.keys()) if all_cities else []

if not cities:
    st.error("No city data available.")
    st.stop()

city = st.selectbox("Select City", cities)
test_days = st.slider("Backtest Days", 7, 14, 14)

st.markdown("---")

# --------------------------
# LOAD DATA
# --------------------------
df_city = all_cities[city]

if len(df_city) < test_days + 30:
    st.error("❌ Not enough data for backtesting.")
    st.stop()


# --------------------------
# PREPARE BACKTEST
# --------------------------
real_tail = df_city.tail(test_days).reset_index(drop=True)

try:
    forecast_df = ensemble_forecast(df_city, city=city, horizon=test_days)
    components = {
        "lstm": forecast_df["yhat_lstm"].values if "yhat_lstm" in forecast_df.columns else forecast_df["final_pred"].values,
        "prophet": forecast_df["yhat_prophet"].values if "yhat_prophet" in forecast_df.columns else forecast_df["final_pred"].values,
        "xgb": forecast_df["yhat_xgb"].values if "yhat_xgb" in forecast_df.columns else forecast_df["final_pred"].values
    }
except Exception as e:
    st.error(f"Error in ensemble forecast: {str(e)}")
    st.stop()


# --------------------------
# BUILD COMPARISON TABLE
# --------------------------
df_eval = pd.DataFrame({
    "date": real_tail["date"],
    "Actual": real_tail["temperature"].values,
    "LSTM": components["lstm"],
    "Prophet": components["prophet"],
    "XGBoost": components["xgb"],
    "Ensemble": forecast_df["final_pred"]
})

st.markdown("### 📋 Evaluation Table")
st.dataframe(df_eval, use_container_width=True)


# --------------------------
# METRICS
# --------------------------
def mae(y, yhat): return round(np.mean(np.abs(y - yhat)), 3)
def rmse(y, yhat): return round(np.sqrt(np.mean((y - yhat)**2)), 3)

metrics = {
    "Model": ["LSTM", "Prophet", "XGBoost", "Ensemble"],
    "MAE": [
        mae(df_eval["Actual"], df_eval["LSTM"]),
        mae(df_eval["Actual"], df_eval["Prophet"]),
        mae(df_eval["Actual"], df_eval["XGBoost"]),
        mae(df_eval["Actual"], df_eval["Ensemble"]),
    ],
    "RMSE": [
        rmse(df_eval["Actual"], df_eval["LSTM"]),
        rmse(df_eval["Actual"], df_eval["Prophet"]),
        rmse(df_eval["Actual"], df_eval["XGBoost"]),
        rmse(df_eval["Actual"], df_eval["Ensemble"]),
    ]
}

df_metrics = pd.DataFrame(metrics)

st.markdown("### 📊 Model Performance Scores")
st.dataframe(df_metrics, use_container_width=True)


# --------------------------
# VISUALIZE PERF
# --------------------------
fig_mae = px.bar(df_metrics, x="Model", y="MAE", text="MAE", title="🔻 Mean Absolute Error")
fig_rmse = px.bar(df_metrics, x="Model", y="RMSE", text="RMSE", title="🎯 RMSE Comparison")

st.plotly_chart(fig_mae, use_container_width=True)
st.plotly_chart(fig_rmse, use_container_width=True)

st.success("✔ Accuracy evaluation complete.")
