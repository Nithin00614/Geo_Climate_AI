import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# ✅ Load environment variables from the project root (.env)
load_dotenv()

# ✅ Get API key from .env file
API_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_weather_data(city_name):
    """
    Fetch current weather data for a given city using the OpenWeather API.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code != 200:
        print("❌ Error fetching data:", data.get("message"))
        return None

    # ✅ Extract and structure relevant fields
    weather_info = {
        "city": city_name,
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "weather": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return weather_info


def save_weather_data(data):
    """
    Save fetched weather data into a CSV file in the project's /data folder.
    Automatically creates the folder if missing.
    """
    # ✅ Always resolve path relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, "weather_data.csv")

    # ✅ Save or append
    df = pd.DataFrame([data])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

    print(f"\n💾 Weather data saved successfully to {filepath}")
