# -----------------------------
# CLIMATE RISK + EXTREME ALERT ENGINE (FULL VERSION)
# -----------------------------
# Save as:
# src/data_engine/climate_risk_full.py
# -----------------------------

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# RISK WEIGHTS (expert tuned)
# -----------------------------
WEIGHTS = {
    "temperature": 0.30,
    "humidity": 0.20,
    "pressure": 0.10,
    "wind_speed": 0.10,
    "aqi": 0.30
}

# -----------------------------
# NORMALIZED COMPONENT RISK
# -----------------------------
def compute_component_risk(temp, hum, pres, wind, aqi):
    temp_risk = min(100, max(0, (temp - 30) * 3))
    hum_risk = min(100, max(0, abs(hum - 60) * 2))
    pres_risk = min(100, abs(pres - 1010) * 1.5)
    wind_risk = min(100, wind * 10)
    aqi_risk = min(100, (aqi / 500) * 100)
    return temp_risk, hum_risk, pres_risk, wind_risk, aqi_risk

# -----------------------------
# FINAL RISK SCORE
# -----------------------------
def calculate_risk_score(row):
    t, h, p, w, a = compute_component_risk(
        row["temperature"], row["humidity"],
        row["pressure"], row["wind_speed"], row["aqi"]
    )
    score = (
        t * WEIGHTS["temperature"] +
        h * WEIGHTS["humidity"] +
        p * WEIGHTS["pressure"] +
        w * WEIGHTS["wind_speed"] +
        a * WEIGHTS["aqi"]
    )
    severity = "Low" if score < 30 else "Medium" if score < 60 else "High"
    return round(score, 2), severity

# -----------------------------
# DAILY RISK SUMMARY
# -----------------------------
def generate_daily_risk_summary(df):
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    summary = df.groupby(["city", "date"]).agg(
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_aqi=("aqi", "mean"),
        avg_wind=("wind_speed", "mean"),
        avg_pressure=("pressure", "mean"),
        avg_risk=("risk", "mean"),
        high_alerts=("severity", lambda x: (x == "High").sum())
    ).reset_index()
    return summary

# -----------------------------
# EXTREME WEATHER ANOMALY DETECTOR
# -----------------------------
def detect_extreme_conditions(df):
    """
    Returns a list of extreme weather alerts.
    """
    alerts = []
    df = df.sort_values("timestamp")

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Temperature spike
        if abs(curr["temperature"] - prev["temperature"]) >= 5:
            alerts.append({
                "type": "Temperature Spike",
                "city": curr["city"],
                "value": curr["temperature"],
                "timestamp": curr["timestamp"],
                "msg": "Sudden temperature change detected."
            })

        # AQI surge
        if curr["aqi"] - prev["aqi"] >= 40:
            alerts.append({
                "type": "AQI Surge",
                "city": curr["city"],
                "value": curr["aqi"],
                "timestamp": curr["timestamp"],
                "msg": "Air quality deteriorated rapidly."
            })

        # Humidity crash
        if prev["humidity"] - curr["humidity"] >= 20:
            alerts.append({
                "type": "Humidity Drop",
                "city": curr["city"],
                "value": curr["humidity"],
                "timestamp": curr["timestamp"],
                "msg": "Sharp drop in humidity detected."
            })

        # Strong wind warning
        if curr["wind_speed"] >= 12:
            alerts.append({
                "type": "High Wind",
                "city": curr["city"],
                "value": curr["wind_speed"],
                "timestamp": curr["timestamp"],
                "msg": "Strong wind detected."
            })

        # Pressure anomaly
        if abs(curr["pressure"] - 1010) >= 25:
            alerts.append({
                "type": "Pressure Anomaly",
                "city": curr["city"],
                "value": curr["pressure"],
                "timestamp": curr["timestamp"],
                "msg": "Unusual atmospheric pressure variations."
            })

    return alerts

# -----------------------------
# RADAR CHART (CITY COMPARISON)
# -----------------------------
def generate_city_radar(df):
    """
    df must contain latest rows for each city.
    """
    if df.empty:
        return None

    latest = df.sort_values("timestamp").groupby("city").tail(1)

    categories = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "aqi",
        "risk"
    ]

    fig = go.Figure()

    for _, row in latest.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories],
            theta=categories,
            fill='toself',
            name=row["city"]
        ))

    fig.update_layout(
        title="🌍 City Risk Comparison Radar",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )

    return fig

# -----------------------------
# HEATMAP SUPPORT
# -----------------------------
def generate_risk_heatmap(df_daily):
    pivot = df_daily.pivot_table(
        values="avg_risk",
        index="city",
        columns="date"
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        labels={"color": "Risk Score"},
        title="🔥 Climate Risk Heatmap"
    )
    return fig
