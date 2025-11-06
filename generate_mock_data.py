import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_mock_weather_data():
    cities = ["Delhi", "Mumbai", "Chennai"]
    start_date = datetime.now() - timedelta(days=60)
    data = []

    for city in cities:
        base_temp = {"Delhi": 30, "Mumbai": 32, "Chennai": 33}[city]
        for i in range(60):
            date = start_date + timedelta(days=i)
            temp = base_temp + np.random.normal(0, 2)  # random small variations
            humidity = np.random.randint(40, 80)
            pressure = np.random.randint(1005, 1020)
            weather = np.random.choice(["clear", "cloudy", "rain", "haze"])
            wind_speed = np.random.uniform(1, 6)

            data.append([city, temp, humidity, pressure, weather, wind_speed, date.strftime("%Y-%m-%d")])

    df = pd.DataFrame(data, columns=["city", "temperature", "humidity", "pressure", "weather", "wind_speed", "date_time"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/weather_data.csv", index=False)
    print("✅ Mock weather data generated and saved to data/weather_data.csv")

if __name__ == "__main__":
    generate_mock_weather_data()
