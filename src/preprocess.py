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

def preprocess_weather_data(df):
    """
    Cleans and preprocesses the weather data for model training and prediction.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df.dropna(inplace=True)
    
    # Ensure required columns exist
    expected_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    scaler = StandardScaler()
    df[expected_cols] = scaler.fit_transform(df[expected_cols])

    return df, scaler
