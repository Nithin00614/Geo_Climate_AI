import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime

DB_PATH = "data/climate_ai.db"
CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Chandigarh": (30.7333, 76.7794),
    "Surat": (21.1702, 72.8311),
    "Madurai": (9.9252, 78.1198)
}


def fetch_historical_data(city, lat, lon, start="2018-01-01", end="2024-12-31"):
    import time
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto"
    }

    print(f"🌍 Fetching historical data for {city}...")
    response = requests.get(url, params=params)

    # If API fails, skip
    if response.status_code != 200:
        print(f"⚠️ API request failed for {city}: {response.status_code}")
        return None

    data = response.json()

    # Check for 'daily' key
    if "daily" not in data:
        print(f"⚠️ No daily data available for {city} (skipping).")
        return None

    # Check if time key exists
    if "time" not in data["daily"]:
        print(f"⚠️ Missing time field for {city} (skipping).")
        return None

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "city": city,
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "rainfall": data["daily"]["precipitation_sum"]
    })

    # Small delay to avoid rate limit
    time.sleep(0.5)
    return df


def store_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("weather_data", conn, if_exists="append", index=False)
    conn.close()

    
def run_pipeline():
    os.makedirs("data", exist_ok=True)

    # Open log file
    log_path = os.path.join("data", "fetch_log.txt")
    with open(log_path, "a", encoding="utf-8") as log:

        print("🌍 Starting data pipeline...\n", file=log)
        print("=" * 50, file=log)

        all_data = []
        for city, (lat, lon) in CITIES.items():
            df = fetch_historical_data(city, lat, lon)
            if df is not None:
                print(f"✅ {city}: {len(df)} rows fetched successfully.", file=log)
                all_data.append(df)
            else:
                print(f"⚠️ {city}: No data returned or skipped.", file=log)

        print("\n🔄 Combining and saving valid data...", file=log)
        all_data = [d for d in all_data if d is not None]

        if not all_data:
            print("❌ No valid data fetched. Please check API or connection.", file=log)
            return

        combined = pd.concat(all_data)
        combined.drop_duplicates(subset=["date", "city"], inplace=True)
        combined.to_csv("data/historical_combined.csv", index=False)

        print(f"✅ Pipeline completed successfully! Total rows: {len(combined)}", file=log)
        print("=" * 50 + "\n", file=log)


if __name__ == "__main__":
    run_pipeline()
