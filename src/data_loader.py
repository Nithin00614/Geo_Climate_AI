from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import datetime

# Load .env file
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_weather_data(cities=None):
    """
    Fetch weather data for multiple cities and save it as a CSV.
    """
    if cities is None:
        cities = ["Mumbai", "Delhi", "Chennai", "Bangalore", "Kolkata", "Hyderabad"]

    all_data = []

    for city in cities:
        print(f"🌍 Fetching weather data for {city}...")
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            weather_info = {
                "city": city,
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_data.append(weather_info)
        else:
            print(f"❌ Failed to fetch data for {city}: {response.json().get('message')}")

    if not all_data:
        print("⚠️ No data fetched. Please check your API key or city names.")
        return None

    df = pd.DataFrame(all_data)

    # Ensure /data folder exists
    os.makedirs("../data", exist_ok=True)
    file_path = "../data/weather_data.csv"

    # Append or create
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print(f"\n✅ Weather data saved to {file_path}")
    return df
