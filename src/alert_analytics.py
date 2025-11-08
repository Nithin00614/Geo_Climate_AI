import os
import pandas as pd
import numpy as np
import plotly.express as px
from alert_logger import load_alert_history
from iot_simulator import CITY_COORDS
from lstm_model import predict_next_7_days

DATA_PATH = "data/alerts_log.txt"

def load_alerts_df():
    """Loads all alerts as DataFrame"""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=["Timestamp", "City", "Type", "Message"])
    df = load_alert_history(limit=10000)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def compute_city_alert_stats(df):
    """Computes number of alerts per city and alert type."""
    if df.empty:
        return pd.DataFrame(columns=["City", "AlertCount", "Heatwave", "Rainfall", "LowHumidity"])

    stats = df.groupby(["City", "Type"]).size().unstack(fill_value=0)
    stats["AlertCount"] = stats.sum(axis=1)
    stats = stats.reset_index()
    return stats

def compute_risk_score(alert_stats, df_forecasts):
    """Combines AI forecast mean + alert frequency into a risk index."""
    risk_rows = []
    for city in alert_stats["City"].unique():
        coords = CITY_COORDS.get(city, (None, None))
        alerts = alert_stats[alert_stats["City"] == city].iloc[0].to_dict()
        total_alerts = alerts.get("AlertCount", 0)
        # fetch mean predicted temperature (if model exists)
        mean_temp = np.nan
        try:
            f = predict_next_7_days(df_forecasts, city)
            if f is not None:
                mean_temp = float(f["predicted_temperature"].mean())
        except Exception:
            mean_temp = np.nan
        # risk formula (weighted)
        if np.isnan(mean_temp):
            risk_score = 0.4 * total_alerts
        else:
            risk_score = 0.6 * (mean_temp / 40) + 0.4 * (min(total_alerts, 50) / 50)
        # normalize risk_score between 0–1
        risk_score = round(min(1, risk_score), 3)

        risk_rows.append({
            "City": city,
            "MeanTemp": mean_temp,
            "TotalAlerts": total_alerts,
            "RiskScore": risk_score,
            "lat": coords[0],
            "lon": coords[1],
        })
    return pd.DataFrame(risk_rows)

def plot_alert_trends(df):
    """Plot alert frequency over time."""
    if df.empty:
        return None
    daily = df.groupby(df["Timestamp"].dt.date).size().reset_index(name="Count")
    fig = px.line(daily, x="Timestamp", y="Count", title="📈 Alerts Over Time", markers=True)
    return fig

def plot_top_cities(stats):
    """Bar chart for top 5 cities with most alerts."""
    if stats.empty:
        return None
    top5 = stats.sort_values("AlertCount", ascending=False).head(5)
    fig = px.bar(top5, x="City", y="AlertCount", color="City", title="🔥 Top 5 Cities with Most Alerts")
    return fig

def plot_alert_type_distribution(df):
    """Pie chart showing alert type distribution."""
    if df.empty:
        return None
    pie = df["Type"].value_counts().reset_index()
    pie.columns = ["Type", "Count"]
    fig = px.pie(pie, values="Count", names="Type", title="🌧️ Alert Type Distribution")
    return fig
