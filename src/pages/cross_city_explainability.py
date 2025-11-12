# src/pages/cross_city_explainability.py
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
from models.shap_explainer import _normalize_city_data, train_xgb_for_shap
import shap

st.set_page_config(page_title="🌍 Cross-City Explainability", layout="wide")
st.title("🌍 GeoClimate AI — Cross-City SHAP Comparison")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    st.warning("⚠️ No /data folder found. Train models first.")
    st.stop()

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not files:
    st.warning("⚠️ No city data files found.")
    st.stop()

# User controls
st.sidebar.header("⚙️ Settings")
sample_size = st.sidebar.slider("Rows per City (for speed)", 20, 200, 60)
show_map = st.sidebar.checkbox("Show City Map View", True)

@st.cache_data(show_spinner=False)
def analyze_all_cities():
    results = []
    for i, f in enumerate(files, 1):
        city = os.path.splitext(f)[0].capitalize()
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            df = _normalize_city_data(df)
            df = df.sample(min(len(df), sample_size), random_state=42)

            # Train lightweight XGB & compute SHAP importance
            model, X, X_scaled = train_xgb_for_shap(df, city)
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)
            shap_abs = np.abs(shap_values.values).mean(axis=0)
            importance = dict(zip(X.columns, shap_abs))
            top_feature = max(importance, key=importance.get)
            results.append({
                "City": city,
                "Top Feature": top_feature.capitalize(),
                "Importance": round(importance[top_feature], 4),
                "Temperature": df["temperature"].mean(),
                "Humidity": df["humidity"].mean(),
                "Rainfall": df["rainfall"].mean(),
            })
            st.write(f"✅ Analyzed {city}")
        except Exception as e:
            st.write(f"⚠️ Skipped {city}: {e}")
    return pd.DataFrame(results)

if st.button("🚀 Run Cross-City Explainability"):
    with st.spinner("Analyzing feature importance across cities..."):
        summary_df = analyze_all_cities()
        if summary_df.empty:
            st.error("No cities could be analyzed.")
            st.stop()

        st.success(f"✅ Analyzed {len(summary_df)} cities.")
        st.dataframe(summary_df, use_container_width=True)

        # Bar chart of Top Feature Importance
        fig = px.bar(
            summary_df.sort_values("Importance", ascending=False),
            x="City", y="Importance",
            color="Top Feature",
            title="Feature Dominance Across Cities",
            text="Top Feature"
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart summary of feature frequency
        freq = summary_df["Top Feature"].value_counts().reset_index()
        freq.columns = ["Feature", "Count"]
        st.subheader("🧠 Feature Dominance Frequency")
        pie = px.pie(freq, values="Count", names="Feature", title="Which Feature Matters Most Across Cities")
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(pie, use_container_width=True)

        # Optional map view (simulated lat/lon)
        if show_map:
            coords = {
                "Bengaluru": (12.97, 77.59), "Delhi": (28.61, 77.21),
                "Mumbai": (19.07, 72.87), "Chennai": (13.08, 80.27),
                "Kolkata": (22.57, 88.36), "Hyderabad": (17.38, 78.48),
                "Pune": (18.52, 73.85), "Ahmedabad": (23.02, 72.57),
                "Jaipur": (26.91, 75.79), "Lucknow": (26.85, 80.95),
            }
            summary_df["lat"] = summary_df["City"].map(lambda c: coords.get(c, (np.nan, np.nan))[0])
            summary_df["lon"] = summary_df["City"].map(lambda c: coords.get(c, (np.nan, np.nan))[1])
            map_df = summary_df.dropna(subset=["lat", "lon"])
            if not map_df.empty:
                st.subheader("🗺️ Feature Dominance Map")
                mfig = px.scatter_mapbox(
                    map_df,
                    lat="lat", lon="lon",
                    color="Top Feature",
                    size="Importance",
                    hover_name="City",
                    mapbox_style="carto-darkmatter",
                    zoom=4.1,
                    title="Dominant Feature by City"
                )
                st.plotly_chart(mfig, use_container_width=True)
