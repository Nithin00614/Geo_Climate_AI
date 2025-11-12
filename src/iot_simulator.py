# src/iot_simulator.py
"""
IoT simulator with SQLite persistence + alert notifications.

Features:
- Simulates sensor readings for multiple cities.
- Computes Climate Risk Index (CRI).
- Writes IoT data and alerts to SQLite DB (data/climate_ai.db).
- Still writes CSVs for backward compatibility.
- Provides run_simulation() generator for real-time dashboards.
"""

import os
import time
import random
import sqlite3
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "climate_ai.db")
IOT_CSV = os.path.join(DATA_DIR, "iot_data.csv")
ALERTS_CSV = os.path.join(DATA_DIR, "alerts.csv")
os.makedirs(DATA_DIR, exist_ok=True)

CITY_COORDS = {
    "Bengaluru": (12.97, 77.59),
    "Mumbai": (19.07, 72.87),
    "Delhi": (28.61, 77.21),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.38, 78.48),
    "Pune": (18.52, 73.85),
    "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95)
}

# --- SQLite Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS iot_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            lat REAL,
            lon REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            aqi INTEGER,
            cri REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            lat REAL,
            lon REAL,
            cri REAL,
            level TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- Notification handler ---
def notify_alert(city, cri, level, message):
    """Basic console + file alert; can later connect to email/SMS APIs."""
    print(f"🚨 [{level}] {city} — CRI={cri}: {message}")
    # Placeholder for email/SMS integration
    # send_email(city, cri, message)
    # send_sms(city, cri, message)

# --- CRI computation ---
def compute_cri(temp_c, humidity_pct, rainfall_mm, aqi=None):
    t = max(0.0, min(1.0, (temp_c - 25.0) / 15.0))
    h = max(0.0, min(1.0, (humidity_pct - 50.0) / 50.0))
    r = max(0.0, min(1.0, rainfall_mm / 100.0))
    a = 0.0
    if aqi is not None:
        a = max(0.0, min(1.0, (aqi - 50) / 200.0))
    cri = 100 * (0.45 * t + 0.25 * h + 0.2 * r + 0.1 * a)
    return round(cri, 2)

# --- Generate readings for one batch ---
def generate_readings(cities=None):
    readings = []
    cities = cities or list(CITY_COORDS.keys())
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for city in cities:
        lat, lon = CITY_COORDS.get(city, (None, None))
        temp = round(random.uniform(28.0, 42.0), 2)
        humidity = round(random.uniform(50.0, 95.0), 2)
        rainfall = round(random.uniform(0.0, 120.0), 2)
        aqi = int(random.uniform(100, 250))
        cri = compute_cri(temp, humidity, rainfall, aqi=aqi)

        row = {
            "timestamp": ts,
            "city": city,
            "lat": lat,
            "lon": lon,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "aqi": aqi,
            "cri": cri
        }
        readings.append(row)
    return readings

# --- Save batch to SQLite & CSV ---
def save_to_storage(readings):
    if not readings:
        return

    df = pd.DataFrame(readings)

    # Write to CSV
    if not os.path.exists(IOT_CSV):
        df.to_csv(IOT_CSV, index=False, mode="w", header=True)
    else:
        df.to_csv(IOT_CSV, index=False, mode="a", header=False)

    # Write to SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("iot_data", conn, if_exists="append", index=False)
    conn.commit()

    # Initialize alerts list BEFORE loop
    alerts = []

    # Handle alerts
    for row in readings:
        if row["cri"] >= 65:
            level = "HIGH" if row["cri"] >= 80 else "MEDIUM"
            message = f"High climate risk detected (CRI={row['cri']})"
            alerts.append({
                "timestamp": row["timestamp"],
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "cri": row["cri"],
                "level": level,
                "message": message
            })
            notify_alert(row["city"], row["cri"], level, message)

    # Save alerts if any
    if alerts:
        adf = pd.DataFrame(alerts)
        if not os.path.exists(ALERTS_CSV):
            adf.to_csv(ALERTS_CSV, index=False, mode="w", header=True)
        else:
            adf.to_csv(ALERTS_CSV, index=False, mode="a", header=False)
        adf.to_sql("alerts", conn, if_exists="append", index=False)
        conn.commit()

    conn.close()

    # ✅ Print summary safely
    print(f"✅ Saved {len(readings)} readings, {len(alerts)} alerts.")


# --- Run simulation (generator) ---
def run_simulation(interval_seconds=5, iterations=None, cities=None, stop_event=None):
    init_db()
    i = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        batch = generate_readings(cities)
        save_to_storage(batch)
        yield batch
        i += 1
        if iterations is not None and i >= iterations:
            break
        time.sleep(interval_seconds)

# --- Maintenance ---
def clear_logs():
    for f in [IOT_CSV, ALERTS_CSV, DB_PATH]:
        if os.path.exists(f):
            os.remove(f)
    print("🧹 Cleared IoT CSV, alerts CSV, and SQLite DB.")


import yagmail
import json
import streamlit as st  # for in-app notification (only active if dashboard open)

# --- Load email config ---
CONFIG_PATH = "src/config_email.json"
EMAIL_CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        EMAIL_CONFIG = json.load(f)

def notify_alert(city, cri, level, message):
    """Enhanced alert notification system."""
    alert_text = f"🚨 [{level}] {city}: {message} (CRI={cri})"
    print(alert_text)

    # Browser notification (for Streamlit UI)
    try:
        st.toast(alert_text)
    except Exception:
        pass  # ignore if Streamlit not active

    # Email notifications for HIGH alerts
    if EMAIL_CONFIG.get("enable_email", False) and level == "HIGH":
        try:
            yag = yagmail.SMTP(
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["sender_password"]
            )
            subject = f"🚨 High Climate Risk Alert: {city}"
            body = f"{message}\n\nCity: {city}\nCRI: {cri}\nLevel: {level}"
            yag.send(EMAIL_CONFIG["receiver_email"], subject, body)
            print(f"📧 Email sent to {EMAIL_CONFIG['receiver_email']}")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")



if __name__ == "__main__":
    print("🔧 Running IoT simulator test for 5 iterations...")
    for batch in run_simulation(interval_seconds=1, iterations=5):
        print(batch)
