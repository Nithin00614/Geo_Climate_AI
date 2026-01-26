import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# Project imports – these exist in your repo
# -------------------------------------------------------------------
from src.utils.data_loader import load_city_data

# List of cities your project tracks (update if needed)
CITY_OPTIONS = [
    "Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Mumbai",
    "Hyderabad", "Kolkata", "Pune", "Lucknow", "Jaipur"
]

# Rough coordinates for Indian cities (for the map)
CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
}

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def make_summary_cards(total_cities, total_rows, last_date):
    """Render top summary section with dark cards (no white bars)."""
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])

    def card(col, title, value, caption):
        col.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:14px 18px;
                border-radius:14px;
                border:1px solid #1f2933;
            ">
              <div style="font-size:11px; color:#9CA3AF; text-transform:uppercase;">
                {title}
              </div>
              <div style="font-size:24px; font-weight:600; color:#F9FAFB;margin-top:4px;">
                {value}
              </div>
              <div style="font-size:11px; color:#6B7280;margin-top:2px;">
                {caption}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    card(c1, "Cities Tracked", str(total_cities), "Active forecast locations")
    card(c2, "Total Data Points", f"{total_rows:,}", "Historical + real weather rows")
    card(
        c3,
        "Last Data Update",
        last_date.strftime("%d %b %Y") if isinstance(last_date, datetime) else str(last_date),
        "Most recent record across all cities",
    )
    card(
        c4,
        "Project Snapshot",
        "GeoClimate AI",
        "LSTM · Prophet · XGBoost · IoT risk analytics",
    )


def load_all_city_summaries():
    """Load basic stats for each city using load_city_data()."""
    rows = []
    for city in CITY_OPTIONS:
        df = load_city_data(city)
        if df is None or df.empty:
            continue
        df = df.sort_values("date")
        rows.append(
            {
                "city": city,
                "rows": len(df),
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "temp_mean": df["temperature"].mean(),
                "temp_last": df["temperature"].iloc[-1],
                "hum_last": df["humidity"].iloc[-1] if "humidity" in df.columns else None,
            }
        )
    return pd.DataFrame(rows)


def load_alert_summary(db_path="data/climate_ai.db", days=7):
    """
    Optional: read recent alerts from SQLite if DB exists and has 'alerts' table.
    Returns (df_alerts or None, error_message or None).
    """
    import sqlite3

    if not os.path.exists(db_path):
        return None, "Database not found."

    try:
        conn = sqlite3.connect(db_path)
        since = (datetime.utcnow() - timedelta(days=days)).isoformat(" ")
        query = """
            SELECT *
            FROM alerts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=[since])
        conn.close()
        if df.empty:
            return None, "No alerts in the last few days."
        return df, None
    except Exception as e:
        return None, f"Alert summary unavailable: {e}"


# -------------------------------------------------------------------
# Page layout
# -------------------------------------------------------------------
st.set_page_config(page_title="Home Dashboard", layout="wide")

# Title + tagline
st.markdown(
    """
    <h1 style="color:#FBBF24; margin-bottom:0;">🏠 GeoClimate AI — Home</h1>
    <p style="color:#9CA3AF; margin-top:4px;">
      Central overview of forecasting, real-world weather data, and climate risk analytics
      for all tracked cities.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -------------------------------------------------------------------
# Load all city data
# -------------------------------------------------------------------
summary = load_all_city_summaries()
if summary.empty:
    st.error(
        "❌ No weather data detected. Please run your real-data fetch / training pipelines first."
    )
    st.stop()

total_cities = summary["city"].nunique()
total_rows = int(summary["rows"].sum())
last_date = summary["last_date"].max()

# Top summary cards (dark theme)
make_summary_cards(total_cities, total_rows, last_date)

st.markdown("---")

# -------------------------------------------------------------------
# Global / regional map
# -------------------------------------------------------------------
st.subheader("📍 Global / Regional City Map")

map_df = summary[["city"]].copy()
map_df["lat"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
map_df["lon"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
map_df = map_df.dropna(subset=["lat", "lon"])

col_map, col_snapshot = st.columns([2.2, 1.3])

with col_map:
    if map_df.empty:
        st.info("Coordinates not available for current cities.")
    else:
        fig_map = px.scatter_geo(
            map_df,
            lat="lat",
            lon="lon",
            hover_name="city",
            projection="natural earth",
        )
        fig_map.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------------------------
# City snapshot & mini-series
# -------------------------------------------------------------------
with col_snapshot:
    st.markdown("#### 🔎 City Snapshot")
    city_choices = summary["city"].tolist()
    snap_city = st.selectbox("Quick select city", options=city_choices, index=0)

    df_snap = load_city_data(snap_city)
    if df_snap is not None and not df_snap.empty:
        df_snap = df_snap.sort_values("date")
        last_row = df_snap.iloc[-1]

        temp_val = last_row.get("temperature", float("nan"))
        hum_val = last_row.get("humidity", None)
        press_val = last_row.get("pressure", None)

        st.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:14px 18px;
                border-radius:14px;
                border:1px solid #1f2933;
                margin-bottom:10px;
            ">
              <div style="font-size:13px; color:#9CA3AF;">{snap_city}</div>
              <div style="font-size:26px; font-weight:600; color:#F9FAFB;margin-top:4px;">
                {temp_val:.2f}°C
              </div>
              <div style="font-size:11px; color:#6B7280;margin-top:4px;">
                Latest recorded temperature
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_s1, c_s2 = st.columns(2)
        c_s1.metric("Humidity", f"{hum_val:.1f}%" if hum_val is not None else "—")
        c_s2.metric("Pressure", f"{press_val:.0f} hPa" if press_val is not None else "—")

        # mini time-series (last 30 days)
        tail = df_snap.tail(30)
        if len(tail) >= 3:
            fig_mini = px.line(
                tail,
                x="date",
                y="temperature",
                labels={"date": "Date", "temperature": "Temp (°C)"},
                title="Last 30 days",
            )
            fig_mini.update_layout(
                template="plotly_dark",
                height=220,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.caption("Not enough data for last-30-day chart.")
    else:
        st.info("No data for selected city.")

st.markdown("---")

# -------------------------------------------------------------------
# Data coverage across cities
# -------------------------------------------------------------------
st.subheader("📊 Data Coverage Across Cities")

cov_df = summary.sort_values("rows", ascending=False)
fig_cov = px.bar(
    cov_df,
    x="city",
    y="rows",
    text="rows",
    labels={"city": "City", "rows": "Rows (data points)"},
)
fig_cov.update_traces(textposition="outside")
fig_cov.update_layout(
    template="plotly_dark",
    height=360,
    yaxis_title="Number of rows",
)
st.plotly_chart(fig_cov, use_container_width=True)

# -------------------------------------------------------------------
# Optional: recent climate / IoT alerts overview
# -------------------------------------------------------------------
st.subheader("⚠️ Recent Climate / IoT Alerts (Last 7 Days)")

alerts_df, alerts_msg = load_alert_summary()
if alerts_df is None:
    # show a friendly note only (no crash)
    st.caption(alerts_msg or "No alerts available.")
else:
    # quick stats: high / medium / low
    if "severity" in alerts_df.columns:
        sev_counts = alerts_df["severity"].value_counts().to_dict()
        c_a1, c_a2, c_a3 = st.columns(3)
        c_a1.metric("High", sev_counts.get("High", 0))
        c_a2.metric("Medium", sev_counts.get("Medium", 0))
        c_a3.metric("Low", sev_counts.get("Low", 0))

    # show a small table of latest alerts
    latest = alerts_df.head(10).copy()
    for col in ("timestamp", "city", "alert_type", "severity", "message"):
        if col not in latest.columns:
            latest[col] = ""
    latest = latest[["timestamp", "city", "alert_type", "severity", "message"]]
    st.dataframe(latest, use_container_width=True, height=260)

st.markdown("---")

# -------------------------------------------------------------------
# Quick links / actions section (no external dependency)
# -------------------------------------------------------------------
st.subheader("🚀 Quick Actions")

c_q1, c_q2, c_q3 = st.columns(3)
c_q1.write("🔮 Open 14-Day Forecast page from the sidebar to inspect city-level predictions.")
c_q2.write("🧠 Use **Model Accuracy** & **Ensemble Forecast Insights** pages for diagnostics.")
c_q3.write("📬 Daily summary emails & IoT alerts are managed by your scheduler / scripts.")
