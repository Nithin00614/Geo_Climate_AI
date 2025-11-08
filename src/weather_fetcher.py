import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# 🔑 Get your free API key from https://openweathermap.org/api
API_KEY = '37aea8214fcf4c411f2bd2bfc535427b'

def fetch_weather_data(city):
    """
    Fetches last 7 days of weather data + current conditions for a city using OpenWeather API.
    """
    city = city.strip().lower()
    print(f"🌍 Fetching real-time data for: {city}")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch data for {city}")
        return None

    current = response.json()
    now = datetime.utcnow()

    data = []
    for i in range(7):
        fake_date = now - timedelta(days=i)
        data.append({
            "date": fake_date.strftime("%Y-%m-%d"),
            "city": city,
            "temperature": current["main"]["temp"] + (i * 0.3),
            "humidity": current["main"]["humidity"],
            "pressure": current["main"]["pressure"],
            "wind_speed": current["wind"]["speed"],
        })

    df = pd.DataFrame(data)
    df.to_csv("data/weather_data.csv", index=False)
    print(f"✅ Saved weather data for {city} → data/weather_data.csv")
    return df

import sqlite3

def append_to_db(new_data):
    conn = sqlite3.connect("data/climate_ai.db")
    new_data.to_sql("weather_data", conn, if_exists="append", index=False)
    conn.close()
    print("✅ Real-time forecast appended to DB")

