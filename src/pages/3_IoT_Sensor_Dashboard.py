# src/pages/3_IoT_Sensor_Dashboard.py
"""
IoT Sensor Dashboard (Read-only)
Reads IoT sensor logs from data/climate_ai.db -> table iot_data
Displays KPIs, time-series charts, alert summaries, map, daily summary and CSV export.
Replace your existing IoT dashboard page with this file.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# Config
# -------------------------
DB_PATH = Path("data/climate_ai.db")
TABLE_NAME = "iot_data"

st.set_page_config(
    page_title="IoT Sensor Dashboard",
    layout="wide",
)

# -------------------------
# Utilities
# -------------------------
@st.cache_data(ttl=60)
def load_iot_table(db_path: Path, table_name: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(db_path)) as conn:
        # Check table exists
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        cur = conn.execute(q, (table_name,))
        if cur.fetchone() is None:
            return pd.DataFrame()
        # Read table
        df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY timestamp DESC", conn)
    # Normalize columns & types
    if "timestamp" in df.columns:
        # ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # try to guess column
        df["timestamp"] = pd.NaT
    # numeric conversions
    for col in ["temperature", "humidity", "pressure", "wind_speed", "aqi", "risk"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # fill missing severity with "Unknown"
    if "severity" in df.columns:
        df["severity"] = df["severity"].fillna("Unknown")
    else:
        df["severity"] = "Unknown"
    # ensure city column exists
    if "city" not in df.columns:
        df["city"] = "unknown"
    return df

def safe_format(x):
    try:
        if pd.isna(x):
            return "—"
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)
    except Exception:
        return str(x)

def download_link_df(df: pd.DataFrame, filename: str):
    """
    Show a Streamlit download button for the given dataframe.
    Uses the correct Streamlit kwarg `file_name`.
    The key is made unique by including the filename.
    """
    return st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,         # <- correct kwarg name
        mime="text/csv",
        key=f"dl_{filename}"
    )


# -------------------------
# Load data
# -------------------------
st.title("📡 IoT Sensor Dashboard — Data Viewer & Analytics")
st.markdown("Read-only dashboard: visualise IoT readings, alerts, risk and export data.")

df = load_iot_table(DB_PATH, TABLE_NAME)

if df.empty:
    st.warning(
        f"No IoT data found. Expected table `{TABLE_NAME}` in `{DB_PATH}`.\n\n"
        "If you're testing locally and want a fresh DB, either run the simulator to create the table "
        "or delete the old DB and let the pipeline recreate it."
    )
    st.stop()

# -------------------------
# Controls: filters
# -------------------------
unique_cities = sorted(df["city"].dropna().unique().tolist())
default_city = unique_cities[0] if unique_cities else "all"

with st.sidebar:
    st.header("Controls")
    city_filter = st.selectbox("City", ["All"] + unique_cities, index=0, key="iot_city_filter")
    max_days = st.slider("Show last N days", min_value=1, max_value=90, value=14, step=1)
    start_date = st.date_input("From", value=(datetime.utcnow() - timedelta(days=max_days)).date(), key="iot_start")
    end_date = st.date_input("To", value=datetime.utcnow().date(), key="iot_end")

# apply filters
mask = pd.Series(True, index=df.index)
if city_filter != "All":
    mask &= df["city"].str.lower() == city_filter.lower()
if "timestamp" in df.columns:
    mask &= (df["timestamp"].dt.date >= pd.to_datetime(start_date).date()) & (df["timestamp"].dt.date <= pd.to_datetime(end_date).date())
df_f = df.loc[mask].copy()

if df_f.empty:
    st.info("No records match the selected filters. Adjust filters to see data.")
    st.stop()

# -------------------------
# Header KPIs (latest row)
# -------------------------
latest = df_f.sort_values("timestamp", ascending=False).iloc[0]

k1, k2, k3, k4, k5 = st.columns([1.5, 1.2, 1.2, 1.2, 1.2])
with k1:
    st.metric("🔎 Latest reading (city)", f"{latest.get('city', '—')}", safe_format(latest.get("timestamp")))
with k2:
    st.metric("🌡️ Temperature (°C)", safe_format(latest.get("temperature")))
with k3:
    st.metric("💧 Humidity (%)", safe_format(latest.get("humidity")))
with k4:
    st.metric("🌫️ AQI", safe_format(latest.get("aqi")))
with k5:
    st.metric("⚠️ Risk", safe_format(latest.get("risk")))

# -------------------------
# Recent table + export
# -------------------------
st.subheader("📋 Recent IoT Records")
st.caption(f"Showing latest {len(df_f)} rows matching filters")

# small housekeeping for display
display_cols = ["timestamp", "city", "temperature", "humidity", "pressure", "wind_speed", "aqi", "risk", "severity"]
table_cols = [c for c in display_cols if c in df_f.columns]
df_show = df_f[table_cols].sort_values("timestamp", ascending=False).reset_index(drop=True)

st.dataframe(df_show, use_container_width=True, height=220, key="iot_table")

download_link_df(df_show, filename=f"iot_data_{city_filter}_{start_date}_to_{end_date}.csv")

# -------------------------
# Alerts summary & severity
# -------------------------
st.subheader("🚨 Alerts Summary")
col_a, col_b = st.columns(2)
with col_a:
    # count high/medium/low in severity
    sev_counts = df_f["severity"].value_counts().to_dict()
    st.markdown("**Severity counts**")
    st.write(pd.Series(sev_counts).rename("count"))
with col_b:
    # Top latest alerts (non-Unknown)
    top_alerts = df_f[df_f["severity"].str.lower() != "unknown"].sort_values("timestamp", ascending=False).head(10)
    st.markdown("**Latest alerts (top 10)**")
    if top_alerts.empty:
        st.info("No recent alerts.")
    else:
        st.table(top_alerts[["timestamp", "city", "severity", "aqi", "risk"]].head(10).rename(columns={"timestamp": "time"}))

# -------------------------
# Time-series charts
# -------------------------
st.subheader("📈 Time-series")

ts_tab1, ts_tab2, ts_tab3 = st.tabs(["Temperature", "Humidity", "AQI"])
with ts_tab1:
    if "temperature" in df_f.columns:
        fig_t = px.line(df_f.sort_values("timestamp"), x="timestamp", y="temperature",
                        title="Temperature over time", labels={"timestamp": "Time", "temperature": "°C"})
        st.plotly_chart(fig_t, use_container_width=True, key="plot_temp")
    else:
        st.info("Temperature column missing.")

with ts_tab2:
    if "humidity" in df_f.columns:
        fig_h = px.line(df_f.sort_values("timestamp"), x="timestamp", y="humidity",
                        title="Humidity over time", labels={"timestamp": "Time", "humidity": "%"})
        st.plotly_chart(fig_h, use_container_width=True, key="plot_hum")
    else:
        st.info("Humidity column missing.")

with ts_tab3:
    if "aqi" in df_f.columns:
        fig_a = px.line(df_f.sort_values("timestamp"), x="timestamp", y="aqi",
                        title="AQI over time", labels={"timestamp": "Time", "aqi": "AQI"})
        st.plotly_chart(fig_a, use_container_width=True, key="plot_aqi")
    else:
        st.info("AQI column missing.")

# -------------------------
# Daily aggregates
# -------------------------
st.subheader("📊 Daily Summary")

df_days = df_f.copy()
df_days["date"] = df_days["timestamp"].dt.date
agg = df_days.groupby("date").agg(
    avg_temp=("temperature", "mean"),
    avg_humidity=("humidity", "mean"),
    avg_aqi=("aqi", "mean"),
    alerts=("severity", lambda s: (s.str.lower() != "unknown").sum())
).reset_index().sort_values("date")

if not agg.empty:
    fig_daily = px.bar(agg, x="date", y=["avg_temp", "avg_humidity", "avg_aqi"],
                       title="Daily averages (temp / humidity / aqi)", barmode="group")
    st.plotly_chart(fig_daily, use_container_width=True, key="plot_daily")
    st.dataframe(agg, use_container_width=True, height=220, key="daily_table")
else:
    st.info("Not enough data for daily summary.")

# -------------------------
# Geographical view (if lat/lon exist)
# -------------------------
st.subheader("🗺️ Sensor Locations Map")
if "lat" in df_f.columns and "lon" in df_f.columns:
    df_map = df_f.dropna(subset=["lat", "lon"]).sort_values("timestamp", ascending=False).drop_duplicates(subset=["city"])
    if df_map.empty:
        st.info("No geolocation entries found.")
    else:
        # scatter_geo to avoid mapbox token requirements
        fig_map = px.scatter_geo(df_map, lat="lat", lon="lon", hover_name="city",
                                 size="aqi" if "aqi" in df_map.columns else None,
                                 title="Latest known sensor locations")
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True, key="plot_map")
else:
    st.info("No lat/lon columns found in table — map disabled.")

# -------------------------
# Quick analytics cards (bottom)
# -------------------------
st.subheader("🔎 Quick Analytics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Avg Temp", f"{agg['avg_temp'].mean():.2f} °C" if not agg.empty else "—")
with c2:
    st.metric("Avg Humidity", f"{agg['avg_humidity'].mean():.2f} %" if not agg.empty else "—")
with c3:
    st.metric("Avg AQI", f"{agg['avg_aqi'].mean():.2f}" if not agg.empty else "—")
with c4:
    st.metric("High Alerts (total)", int(df_f[df_f["severity"].str.lower() == "high"].shape[0]) if "severity" in df_f.columns else 0)

# -------------------------
# Footer: quick export & notes
# -------------------------
st.write("---")
st.markdown(
    """
**Notes & tips**
- This page is read-only — to generate more IoT rows use your simulator script (outside of this page).
- If the DB schema changed (missing columns or empty table) re-run the simulator or run your DB fix script.
- Use the CSV export to download filtered data for offline analysis.
"""
)
