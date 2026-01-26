# File: src/pages/4_🧭_Climate_Risk_Alerts.py
# Paste/replace this file in your src/pages folder.
# Upgraded Climate Risk Analytics dashboard page.

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Climate Risk Analytics", layout="wide")

# -----------------------
# Config / Defaults
# -----------------------
DATA_DIR = "data"
IOT_CSV = os.path.join(DATA_DIR, "iot_data.csv")
SQLITE_DB = os.path.join(DATA_DIR, "climate_ai.db")

# default city list - modify if you have a different set
ALL_CITIES_DEFAULT = [
    "ahmedabad", "bengaluru", "chennai", "delhi", "hyderabad",
    "jaipur", "kolkata", "lucknow", "mumbai", "pune"
]

# risk thresholds: tuned for demonstration, adapt as required
RISK_THRESHOLDS = {
    "low": 0,
    "medium": 30,
    "high": 60
}

# -----------------------
# Utilities
# -----------------------
def load_iot_csv(path):
    return pd.read_csv(path)

def load_iot_sqlite(path):
    # try to find 'iot_data' table and read
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM iot_data", conn)
    finally:
        conn.close()
    return df

def load_iot_data():
    """Try loading IoT data from CSV, else sqlite. Returns DataFrame."""
    if os.path.exists(IOT_CSV):
        df = load_iot_csv(IOT_CSV)
        st.debug = getattr(st, "debug", lambda *a, **k: None)
        return df
    if os.path.exists(SQLITE_DB):
        try:
            df = load_iot_sqlite(SQLITE_DB)
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def ensure_columns(df):
    """Lowercase column names and check required fields. Return cleaned df."""
    df = df.copy()
    # unify column names
    df.columns = [c.lower().strip() for c in df.columns]
    # ensure date column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        # fallback: create synthetic dates (today) if missing
        df["date"] = pd.to_datetime(datetime.utcnow().date())
    # ensure numeric columns
    for col in ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "aqi"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # ensure city
    if "city" not in df.columns:
        df["city"] = "unknown"
    df["city"] = df["city"].astype(str).str.lower()
    return df

def generate_synthetic_iot(cities, days=14, seed=42):
    """Create synthetic IoT readings for missing cities for the past `days` days."""
    rng = np.random.RandomState(seed)
    rows = []
    today = datetime.utcnow().date()
    for city in cities:
        for d in range(days):
            dt = today - timedelta(days=d)
            # create plausible ranges, could be tuned per-city
            temp = float(rng.normal(loc=28, scale=4))
            humidity = float(np.clip(rng.normal(loc=55, scale=15), 5, 100))
            pressure = float(np.clip(rng.normal(loc=1010, scale=8), 900, 1050))
            wind = float(np.clip(rng.normal(loc=2.5, scale=1.5), 0, 12))
            rainfall = float(max(0.0, rng.normal(loc=2, scale=5)))
            aqi = float(np.clip(rng.normal(loc=100, scale=80), 10, 500))
            rows.append({
                "timestamp": pd.Timestamp(dt),
                "date": dt,
                "city": city,
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "pressure": round(pressure, 2),
                "wind_speed": round(wind, 2),
                "rainfall": round(rainfall, 2),
                "aqi": round(aqi, 0)
            })
    return pd.DataFrame(rows)

def compute_cri(row):
    """
    Compute a simple Climate Risk Index:
      - higher temperature deviation from comfortable range increases risk
      - low humidity increases risk for heat stress/dryness
      - high AQI increases risk
      - high rainfall can increase risk (flood) but we scale gently
    This function returns a 0-100-ish score (not probabilistic).
    """
    score = 0.0
    # temperature: comfortable ~ 20-30 C
    if not pd.isna(row["temperature"]):
        temp = row["temperature"]
        # penalize deviations from 24-28
        score += max(0, abs(temp - 26) * 2.5)
    # humidity: low humidity <30 increases risk for dryness
    if not pd.isna(row["humidity"]):
        humidity = row["humidity"]
        if humidity < 30:
            score += (30 - humidity) * 0.5
    # AQI: linear contribution (0 good -> 500 bad)
    if not pd.isna(row["aqi"]):
        aqi = row["aqi"]
        score += (aqi / 10.0)
    # rainfall: heavy rainfall adds to risk (flood)
    if not pd.isna(row["rainfall"]):
        rain = row["rainfall"]
        score += min(rain * 1.0, 20.0)
    # wind: very strong winds add some risk
    if not pd.isna(row["wind_speed"]):
        wind = row["wind_speed"]
        if wind > 10:
            score += (wind - 10) * 1.5
    return round(score, 2)

def risk_severity(score):
    if score >= RISK_THRESHOLDS["high"]:
        return "High"
    if score >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"

# -----------------------
# Page UI - sidebar filters
# -----------------------
st.title("🚨 Climate Risk Analytics")
st.markdown("Interactive analytics combining IoT sensor data, risk scoring, and alerts.")

with st.sidebar:
    st.header("Filter Data")
    city_filter = st.selectbox("City", options=["All"] + ALL_CITIES_DEFAULT, index=0)
    days = st.slider("Show last N days", min_value=1, max_value=60, value=14)
    use_synthetic = st.checkbox("Fill missing cities with synthetic data (preview)", value=True)
    synthetic_days = st.slider("Synthetic days (if enabled)", 7, 30, 14)
    st.markdown("---")
    st.markdown("⚙️ **Options**")
    show_heatmap = st.checkbox("Show alerts heatmap", value=True)
    show_radar = st.checkbox("Show city radar comparison", value=True)
    download_all = st.button("Export current selection (CSV)")

# -----------------------
# Load data
# -----------------------
raw_df = load_iot_data()
if raw_df.empty:
    st.warning("No IoT data found in data/iot_data.csv or data/climate_ai.db. You can enable synthetic fill to preview analytics.")
df = ensure_columns(raw_df)

# ensure all cities list: prefer a local cities.csv if present
cities_file = os.path.join(DATA_DIR, "cities.txt")
if os.path.exists(cities_file):
    try:
        with open(cities_file) as f:
            file_cities = [l.strip().lower() for l in f.readlines() if l.strip()]
            ALL_CITIES = file_cities
    except Exception:
        ALL_CITIES = ALL_CITIES_DEFAULT
else:
    ALL_CITIES = ALL_CITIES_DEFAULT

# If user wants synthetic fill - create synthetic data for missing cities
if use_synthetic:
    present_cities = sorted(df["city"].dropna().unique().tolist())
    missing = [c for c in ALL_CITIES if c not in present_cities]
    if missing:
        synth = generate_synthetic_iot(missing, days=synthetic_days)
        # append only dates within requested range
        df = pd.concat([df, synth], ignore_index=True, sort=False)

# Filter last N days
cutoff = (datetime.utcnow().date() - timedelta(days=days-1))
df["date"] = pd.to_datetime(df["date"]).dt.date
df = df[df["date"] >= cutoff]

# Filter by city if requested
if city_filter != "All":
    df = df[df["city"] == city_filter.lower()]

# Compute CRI and severity
df["cri"] = df.apply(compute_cri, axis=1)
df["severity"] = df["cri"].apply(risk_severity)

# Mark a 'high_alert' flag for quick counting
df["high_alert"] = np.where(df["severity"] == "High", 1, 0)

# -----------------------
# Aggregate tables for charts
# -----------------------

# Make sure every ALL_CITIES present in the groupings (reindex)
group_high_alerts = df.groupby("city")["high_alert"].sum().reindex(ALL_CITIES, fill_value=0).reset_index()
group_high_alerts.columns = ["city", "high_alerts"]

# Daily avg risk by city - make full matrix with reindex
daily_risk = df.groupby(["date", "city"])["cri"].mean().reset_index()
if not daily_risk.empty:
    daily_pivot = daily_risk.pivot(index="date", columns="city", values="cri").reindex(columns=ALL_CITIES, fill_value=0)
    daily_pivot = daily_pivot.sort_index()
else:
    daily_pivot = pd.DataFrame(index=pd.date_range(datetime.utcnow().date() - timedelta(days=days-1), periods=days).date)
    for c in ALL_CITIES:
        daily_pivot[c] = 0.0
    daily_pivot.index.name = "date"

# Latest snapshot (most recent reading per city)
latest = df.sort_values("timestamp").groupby("city").tail(1).set_index("city")
# ensure all cities present in latest (fill with NaNs)
latest = latest.reindex(ALL_CITIES).reset_index()

# Alerts timeline
alerts_table = df[df["high_alert"] == 1].sort_values(["date", "timestamp"], ascending=False)

# -----------------------
# Top metrics
# -----------------------
st.markdown("### 📊 High Alert Summary")
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    total_high = int(group_high_alerts["high_alerts"].sum())
    st.metric("Total High Alerts (selected range)", f"{total_high}")
with col2:
    avg_risk = round(df["cri"].mean() if not df["cri"].isna().all() else 0, 2)
    st.metric("Avg CRI", f"{avg_risk}")
with col3:
    avg_aqi = round(df["aqi"].mean(skipna=True) if "aqi" in df.columns else 0, 2)
    st.metric("Avg AQI", f"{avg_aqi}")
with col4:
    cities_active = df["city"].nunique()
    st.metric("Cities (with data)", f"{cities_active} / {len(ALL_CITIES)}")

st.markdown("---")

# -----------------------
# High alerts bar chart
# -----------------------
fig_bar = px.bar(
    group_high_alerts,
    x="city",
    y="high_alerts",
    title="Number of High Alerts per City",
    labels={"high_alerts": "High alerts", "city": "City"},
    text="high_alerts",
)
fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_dark", bargap=0.2)
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True, key="cri_bar")

# -----------------------
# City radar comparison (avg metrics)
# -----------------------
if show_radar:
    # compute normalized metrics per city for radar
    agg = df.groupby("city").agg({
        "temperature": "mean",
        "humidity": "mean",
        "aqi": "mean",
        "rainfall": "mean",
        "cri": "mean"
    }).reindex(ALL_CITIES).fillna(0)

    # Normalize each metric to 0-1 for fair radar scale
    agg_norm = (agg - agg.min()) / (agg.max() - agg.min()).replace(0, 1)
    # Build radar traces for a few top cities (or all)
    fig_rad = go.Figure()
    for c in ALL_CITIES:
        values = agg_norm.loc[c].tolist()
        # radar needs closed shape
        fig_rad.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=list(agg_norm.columns) + [agg_norm.columns[0]],
            fill="toself",
            name=c.capitalize()
        ))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, template="plotly_dark", title="City Comparison Radar Chart (normalized)")
    st.plotly_chart(fig_rad, use_container_width=True, key="cri_radar")

# -----------------------
# Heatmap: daily high-alert counts by city
# -----------------------
if show_heatmap:
    heat = df[df["high_alert"]==1].groupby(["date","city"])["high_alert"].count().reset_index().pivot(index="date", columns="city", values="high_alert").reindex(columns=ALL_CITIES, fill_value=0)
    if heat.empty:
        st.info("No high-alert events to display on heatmap.")
    else:
        fig_heat = px.imshow(heat.T, labels=dict(x="date", y="city", color="High alert count"), x=heat.index.astype(str), y=heat.columns, aspect="auto", title="High-alert heatmap (city vs date)")
        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True, key="cri_heat")

# -----------------------
# Daily CRI Trend
# -----------------------
st.markdown("### 📈 Daily Climate Risk Trend")
if daily_pivot.empty:
    st.info("Daily risk data not available.")
else:
    # choose top cities to plot for readability (or all)
    # plot all, but legend can be filtered by user
    fig_line = px.line(
        daily_pivot.reset_index().melt(id_vars="date", var_name="city", value_name="avg_risk"),
        x="date", y="avg_risk", color="city",
        title="Daily Risk Score Trend",
        labels={"avg_risk": "Avg CRI", "date": "Date"}
    )
    fig_line.update_layout(template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True, key="cri_line")

# -----------------------
# Alerts table (recent high alerts)
# -----------------------
st.markdown("### 🔔 Recent High Alerts")
if alerts_table.empty:
    st.info("No high alerts in the selected range.")
else:
    display_alerts = alerts_table[["timestamp","city","temperature","humidity","aqi","cri","severity"]].copy()
    display_alerts = display_alerts.sort_values("timestamp", ascending=False).reset_index(drop=True)
    st.dataframe(display_alerts, use_container_width=True)

# -----------------------
# Latest Snapshot table
# -----------------------
st.markdown("### 🕒 Latest Snapshot per City")
st.dataframe(latest[["city","timestamp","temperature","humidity","aqi","cri","severity"]].rename(columns=str.capitalize), use_container_width=True)

# -----------------------
# Download / Export
# -----------------------
def to_csv_bytes(df_in):
    csv = df_in.to_csv(index=False)
    return csv.encode("utf-8")

export_df = df.copy()
if download_all:
    st.success("Preparing download...")

# provide download button
csv_bytes = to_csv_bytes(export_df)
st.download_button(
    label="📥 Download filtered data (CSV)",
    data=csv_bytes,
    file_name=f"climate_risk_export_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="download_cri"
)

# also provide alerts-only download
alerts_csv = to_csv_bytes(alerts_table) if not alerts_table.empty else to_csv_bytes(pd.DataFrame())
st.download_button(
    label="📥 Download alerts (CSV)",
    data=alerts_csv,
    file_name=f"climate_alerts_export_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    key="download_alerts"
)

# -----------------------
# Tips and next steps
# -----------------------
st.markdown("---")
st.markdown("### 🛠️ Notes & Next steps")
st.markdown("""
- The synthetic data toggle lets you preview analytics for all configured cities without running the IoT simulator.
- Adjust `RISK_THRESHOLDS` and `compute_cri()` to match your domain needs (health, city planning, flood risk, etc).
- To integrate this with your alerting pipeline: when a new IoT reading arrives, compute `cri` and `severity` and insert into DB; this page will automatically display it after refresh.
- If you want, I can:
  - Add geographical map visualizations (choropleth or scatter_mapbox).
  - Add daily PDF summary generation & email dispatch.
  - Add ML anomaly detection (isolation forest) to auto-detect outliers.
""")

# -----------------------
# End of page
# -----------------------
