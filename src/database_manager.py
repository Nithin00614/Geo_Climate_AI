# src/database_manager.py
import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/climate_ai.db")

# -----------------------------------------------------
# Database setup and helpers
# -----------------------------------------------------
def init_db():
    import sqlite3
    import os

    os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"🗄️ Database initialized at: {os.path.abspath(DB_PATH)}")

    # --- Check if iot_data table has correct schema ---
    try:
        cursor.execute("PRAGMA table_info(iot_data);")
        cols = [row[1] for row in cursor.fetchall()]
        if not {"lat", "lon"}.issubset(set(cols)):
            print("⚠️ Old iot_data schema detected — recreating table with lat/lon support.")
            cursor.execute("DROP TABLE IF EXISTS iot_data;")
    except Exception as e:
        print(f"⚠️ Schema check failed: {e}")

    # --- Create tables ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        date TEXT,
        predicted_temperature REAL,
        model_name TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS iot_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        temperature REAL,
        humidity REAL,
        rainfall REAL,
        lat REAL,
        lon REAL,
        timestamp TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        alert_type TEXT,
        message TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()



# -----------------------------------------------------
# Insert / Save functions
# -----------------------------------------------------
def save_forecast(city, forecast_df, model_name="lstm"):
    """Save AI forecast results to DB."""
    conn = sqlite3.connect(DB_PATH)
    forecast_df = forecast_df.copy()
    forecast_df["city"] = city
    forecast_df["model_name"] = model_name
    forecast_df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    forecast_df[["city", "date", "predicted_temperature", "model_name", "created_at"]].to_sql(
        "forecasts", conn, if_exists="append", index=False)
    conn.close()


def save_iot_batch(iot_df):
    """Save a batch of IoT readings."""
    conn = sqlite3.connect(DB_PATH)
    iot_df.to_sql("iot_data", conn, if_exists="append", index=False)
    conn.close()


def save_alert(city, alert_type, message):
    """Save alert record."""
    conn = sqlite3.connect(DB_PATH)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO alerts (city, alert_type, message, timestamp) VALUES (?, ?, ?, ?)",
                 (city, alert_type, message, ts))
    conn.commit()
    conn.close()


# -----------------------------------------------------
# Query / Read functions
# -----------------------------------------------------
def get_recent_iot(city=None, limit=20):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM iot_data"
    if city:
        query += f" WHERE city='{city}'"
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_recent_alerts(limit=50):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM alerts ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df


def get_city_forecasts(city, limit=20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM forecasts WHERE city='{city}' ORDER BY date DESC LIMIT {limit}", conn)
    conn.close()
    return df
