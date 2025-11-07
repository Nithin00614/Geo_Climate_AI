import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_weather_data():
    np.random.seed(42)

    # 🌍 A wider range of cities
    cities = [
        "Bengaluru", "Chennai", "Mumbai", "Delhi", "Kolkata", "Hyderabad",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Coimbatore",
        "Bhopal", "Indore", "Surat", "Thiruvananthapuram", "Mysuru", "Goa",
        "Nagpur", "Visakhapatnam"
    ]

    # 1 year of daily data for each city
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    all_data = []

    for city in cities:
        # Base weather pattern depending on city
        base_temp = np.random.uniform(18, 35)
        temp_fluct = np.random.uniform(3, 8)
        humidity_base = np.random.uniform(55, 85)
        pressure_base = np.random.uniform(1005, 1020)
        wind_base = np.random.uniform(1.5, 6.0)

        for date in date_range:
            temp = base_temp + np.sin((date.timetuple().tm_yday / 365) * 2 * np.pi) * temp_fluct + np.random.normal(0, 0.5)
            humidity = humidity_base + np.random.normal(0, 5)
            pressure = pressure_base + np.random.normal(0, 1)
            wind_speed = wind_base + np.random.normal(0, 0.3)

            all_data.append({
                "date_time": date.strftime("%Y-%m-%d"),
                "city": city,
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "pressure": round(pressure, 2),
                "wind_speed": round(wind_speed, 2)
            })

    df = pd.DataFrame(all_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/weather_data.csv", index=False)
    print(f"✅ Synthetic weather dataset generated with {len(df)} records for {len(cities)} cities.")

if __name__ == "__main__":
    generate_mock_weather_data()
