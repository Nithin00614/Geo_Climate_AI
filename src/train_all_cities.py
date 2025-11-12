# src/train_all_cities.py
import os
import time
import pandas as pd
from lstm_model import train_lstm_model
from weather_fetcher import fetch_weather_data

# Define directories
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 🌎 Extended city list (you can add more anytime)
CITY_LIST = [
    "Bengaluru", "Ahmedabad", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune",
    "Jaipur", "Lucknow", "Chandigarh", "Indore", "Coimbatore", "Nagpur", "Surat", "Thiruvananthapuram",
    "Patna", "Bhopal", "Visakhapatnam", "Vadodara", "Rajkot", "Nashik", "Amritsar", "Goa",
    "Mysuru", "Noida", "Gurugram", "Varanasi", "Guwahati", "Ranchi", "Shimla", "Dehradun"
]

# --- Utility functions ---
def delete_all_old_files():
    """Deletes all model and data files before retraining."""
    print("🧹 Cleaning up old models and data...\n")

    deleted = 0
    for folder in [MODELS_DIR, DATA_DIR]:
        for f in os.listdir(folder):
            if f.endswith((".keras", ".pkl", ".csv")):
                os.remove(os.path.join(folder, f))
                deleted += 1

    if deleted:
        print(f"✅ Deleted {deleted} old files from /models and /data.\n")
    else:
        print("ℹ️ No existing model or data files found to delete.\n")


def fetch_or_load_city(city):
    """Fetch or load cached weather data for a city."""
    file_path = os.path.join(DATA_DIR, f"{city.lower()}.csv")
    try:
        print(f"🌤 Fetching data for {city}...")
        df = fetch_weather_data(city)

        if df is not None and not df.empty:
            df.to_csv(file_path, index=False)
            print(f"💾 Saved {city} data to {file_path}")
            return df
        else:
            print(f"⚠️ No data returned for {city}")
            return None
    except Exception as e:
        print(f"❌ Error fetching {city}: {e}")
        return None


def retrain_all_cities():
    """Deletes all old models, fetches new data, retrains everything."""
    delete_all_old_files()

    trained, failed = [], []
    print(f"🚀 Starting full retraining for {len(CITY_LIST)} cities...\n")

    for i, city in enumerate(CITY_LIST, 1):
        print(f"[{i}/{len(CITY_LIST)}] Training model for {city}...")

        df = fetch_or_load_city(city)
        if df is None or df.empty:
            print(f"⚠️ Skipping {city} due to empty data.")
            failed.append(city)
            continue

        try:
            model, scaler = train_lstm_model(df, city)
            if model:
                print(f"✅ Model trained and saved for {city}\n")
                trained.append(city)
            else:
                print(f"⚠️ Model training failed for {city}\n")
                failed.append(city)
        except Exception as e:
            print(f"❌ Error training {city}: {e}")
            failed.append(city)

        time.sleep(1.5)  # to avoid hitting API rate limits

    print("\n==================== SUMMARY ====================")
    print(f"✅ Successfully trained models: {len(trained)}")
    print(f"❌ Failed or skipped cities: {len(failed)}")
    if failed:
        print("⚠️ Failed cities:", failed)
    print("=================================================")


if __name__ == "__main__":
    retrain_all_cities()
