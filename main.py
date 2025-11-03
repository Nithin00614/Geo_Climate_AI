from src.data_loader import fetch_weather_data, save_weather_data
from src.preprocess import preprocess_weather_data
from src.model import train_temperature_model
from src.predict import predict_temperature
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("🌍 Welcome to GeoClimate-AI")
    print("-" * 50)

    # Step 1: Fetch and save weather data
    while True:
        city = input("\n🏙️ Enter city name (or type 'done' to finish): ").strip()
        if city.lower() == "done":
            break

        weather_data = fetch_weather_data(city)
        if weather_data:
            save_weather_data(weather_data)
        else:
            print(f"⚠️ Skipping {city} due to missing data.")

    # Step 2: Preprocess collected data
    processed_df = preprocess_weather_data()

    if processed_df is None or processed_df.empty:
        print("⚠️ No data available for training. Please fetch more cities.")
        return

    # Step 3: Train model
    model, mae, r2 = train_temperature_model(processed_df)

    if model is not None:
        print(f"\n✅ Model trained successfully!")
        print(f"📊 Mean Absolute Error: {mae:.2f}")
        print(f"📈 R² Score: {r2:.2f}")
    else:
        print("⚠️ Model training failed.")
        return

    # Step 4: Test saved model for prediction
    print("\n🌡️ Testing saved model for prediction...")
    try:
        predicted_temp = predict_temperature(humidity=60, pressure=1012, wind_speed=3.5)
        print(f"🤖 Predicted Temperature: {predicted_temp:.2f}°C")
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")

    print("\n🎯 All steps completed successfully!")

if __name__ == "__main__":
    main()
