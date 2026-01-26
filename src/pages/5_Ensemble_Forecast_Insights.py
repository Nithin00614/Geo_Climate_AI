import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.models.forecast_engine import ensemble_forecast
from src.utils.data_loader import load_all_city_data


# --------------------------
# PAGE HEADER
# --------------------------
st.markdown("# 🔍 Ensemble Forecast Insights")
st.caption("Model-level breakdown from LSTM, Prophet & XGBoost — compared against unified ensemble output.")


# --------------------------
# USER INPUTS
# --------------------------
all_cities = load_all_city_data()
cities = sorted(all_cities.keys()) if all_cities else []

if not cities:
    st.error("No city data available. Check data/real_weather/ folder.")
    st.stop()

city = st.selectbox("Select City", cities)
horizon = st.slider("Forecast Horizon", 7, 14, 14)


st.markdown("---")

# --------------------------
# LOAD DATA + RUN ENSEMBLE
# --------------------------
df_city = all_cities[city]

if df_city is None or df_city.empty:
    st.error(f"No data available for {city}")
    st.stop()

try:
    forecast_df = ensemble_forecast(df_city, city=city, horizon=horizon)
    weights = {
        "lstm": forecast_df["lstm_weight"].iloc[0] if "lstm_weight" in forecast_df.columns else 0.33,
        "prophet": forecast_df["prophet_weight"].iloc[0] if "prophet_weight" in forecast_df.columns else 0.33,
        "xgb": forecast_df["xgb_weight"].iloc[0] if "xgb_weight" in forecast_df.columns else 0.34
    }
    components = {
        "lstm": forecast_df["yhat_lstm"].values if "yhat_lstm" in forecast_df.columns else forecast_df["final_pred"].values,
        "prophet": forecast_df["yhat_prophet"].values if "yhat_prophet" in forecast_df.columns else forecast_df["final_pred"].values,
        "xgb": forecast_df["yhat_xgb"].values if "yhat_xgb" in forecast_df.columns else forecast_df["final_pred"].values
    }
except Exception as e:
    st.error(f"Error generating forecast: {str(e)}")
    st.stop()


# --------------------------
# MODEL WEIGHT DISTRIBUTION
# --------------------------
st.markdown("### ⚖️ Ensemble Weight Distribution")

w_df = pd.DataFrame({
    "Model": ["LSTM", "Prophet", "XGBoost"],
    "Weight (%)": [
        round(weights.get("lstm", 0) * 100, 2),
        round(weights.get("prophet", 0) * 100, 2),
        round(weights.get("xgb", 0) * 100, 2),
    ]
})

fig_w = px.bar(
    w_df, x="Model", y="Weight (%)", color="Model",
    title="Model Contribution to Final Forecast",
    text="Weight (%)"
)

st.plotly_chart(fig_w, use_container_width=True)


st.markdown("---")


# --------------------------
# COMPONENT FORECAST VISUAL
# --------------------------
st.markdown(f"### 📈 Forecast Component Breakdown — {city}")

plot_df = pd.DataFrame({
    "date": forecast_df["date"],
    "Ensemble": forecast_df["final_pred"],
    "LSTM": components["lstm"],
    "Prophet": components["prophet"],
    "XGBoost": components["xgb"]
})

plot_df = plot_df.melt(id_vars=["date"], var_name="Model", value_name="Temperature")

fig_line = px.line(
    plot_df,
    x="date", y="Temperature", color="Model",
    markers=True,
    title=f"📌 Forecast Comparison — Model vs Ensemble"
)

st.plotly_chart(fig_line, use_container_width=True)

st.success("✔ Insights generated successfully.")
