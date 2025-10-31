import sys
import os

# --- Add src folder dynamically to Python path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Now import from src files ---
from data_loader import fetch_weather_data, save_weather_data


def main():
    print("🌦️ GeoClimate-AI: Weather Data Collector")
    city = input("Enter city name: ")
    data = fetch_weather_data(city)

    if data:
        print("\n✅ Weather Data:")
        for k, v in data.items():
            print(f"{k}: {v}")

        save_weather_data(data)


if __name__ == "__main__":
    main()
