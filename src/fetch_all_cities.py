import pandas as pd
from weather_fetcher import fetch_weather_data
import os

def fetch_all_cities():
    cities = [
        "bengaluru", "mumbai", "delhi", "chennai", "hyderabad",
        "pune", "kolkata", "ahmedabad", "coimbatore", "lucknow",
        "jaipur", "bhopal", "surat", "indore", "kochi",
        "vizag", "nagpur", "chandigarh", "patna", "guwahati"
    ]

    all_data = []
    os.makedirs("data", exist_ok=True)

    for city in cities:
        print(f"🌤️ Fetching data for {city}...")
        df = fetch_weather_data(city)
        if df is not None:
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data)
        combined.to_csv("data/weather_data.csv", index=False)
        print("✅ Saved combined weather data for 20 cities.")
    else:
        print("⚠️ No data fetched.")

if __name__ == "__main__":
    fetch_all_cities()
