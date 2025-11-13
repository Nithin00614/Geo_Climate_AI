import pandas as pd
import os

DATA_FOLDER = "data"

# Columns ensemble model expects — but we auto-fill missing ones
REQUIRED_COLS = ["temperature", "humidity", "wind_speed", "rainfall"]

def load_city_data(city: str):
    """
    Smart loader: loads city CSV and auto-fills missing columns.
    Prevents crashes due to missing rainfall/humidity/etc.
    """
    city_file = f"{city.lower()}.csv"
    file_path = os.path.join(DATA_FOLDER, city_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ No data found for city '{city}'. Expected file: {file_path}"
        )

    df = pd.read_csv(file_path)

    # ---- Ensure DATE exists ----
    if "date" not in df.columns:
        raise ValueError(f"❌ '{city}' CSV missing required column: 'date'")

    df["date"] = pd.to_datetime(df["date"])

    # ---- Auto-fill missing columns ----
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' missing in {city}. Auto-filling with 0.")
            df[col] = 0

    return df
