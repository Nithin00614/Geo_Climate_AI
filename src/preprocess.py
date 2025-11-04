import pandas as pd
import os

def preprocess_data(df):
    """
    Clean and preprocess the weather dataset.
    Removes duplicates, handles missing values, and ensures valid numeric types.
    """
    if df is None or df.empty:
        print("⚠️ No data available for preprocessing.")
        return None

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna(subset=["temperature", "humidity", "pressure", "wind_speed"])

    # Ensure numeric types
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")

    # Drop invalid rows again after conversion
    df = df.dropna()

    # Save cleaned version to /data/weather_data_clean.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_path = os.path.join(base_dir, "data", "weather_data_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"✅ Cleaned data saved at: {clean_path}")

    return df
