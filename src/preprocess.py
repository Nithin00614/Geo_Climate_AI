import pandas as pd
import os

def get_data_path(filename="weather_data.csv"):
    """
    Get the absolute path for any file inside the /data directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

def preprocess_weather_data():
    """
    Load, clean, and preprocess weather data for model training.
    """
    input_path = get_data_path("weather_data.csv")
    output_path = get_data_path("weather_data_clean.csv")

    if not os.path.exists(input_path):
        print("❌ No data file found! Run the fetch step first.")
        return None

    df = pd.read_csv(input_path)

    if df.empty:
        print("⚠️ The data file is empty.")
        return None

    print(f"\n📂 Loaded {len(df)} records from {input_path}")

    # Drop duplicates
    df = df.drop_duplicates(subset=["city", "date_time"], keep="last")

    # Drop rows with missing values in essential columns
    df = df.dropna(subset=["temperature", "humidity", "pressure", "wind_speed"])

    # Convert date_time to datetime type if available
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Sort data by date
    df = df.sort_values(by="date_time", ascending=False)

    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved successfully at: {output_path}")

    return df

if __name__ == "__main__":
    preprocess_weather_data()
