import streamlit as st
import plotly.express as px
from utils.sidebar_style import apply_sidebar_style

import sys
sys.path.append("src")

from data_loader import load_city_data
from models.ensemble_forecast import ensemble_forecast

st.set_page_config(page_title="📈 Weather Forecast", layout="wide")
apply_sidebar_style()

st.markdown("# 📈 14-Day Unified Weather Forecast")

city = st.selectbox("Select a city", ["ahmedabad", "mumbai", "delhi", "chennai", "hyderabad", "bangalore"])

if st.button("Generate Forecast"):
    with st.spinner("Generating AI-powered forecast..."):
        df = load_city_data(city)
        forecast_df = ensemble_forecast(df, city)

    st.success("Forecast generated successfully!")

    # Line chart
    fig = px.line(
        forecast_df,
        x="date",
        y="predicted_temperature",
        title=f"14-Day Forecast for {city.capitalize()}",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.dataframe(forecast_df)
