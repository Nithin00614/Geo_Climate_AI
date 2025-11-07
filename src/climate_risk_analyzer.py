import numpy as np
import pandas as pd

def analyze_climate_risk(df, city_name):
    """
    Analyze climate risks based on temperature, humidity, and pressure data.
    Returns a dictionary with interpreted risk levels.
    """

    if df is None or df.empty:
        return {"error": "No data available for risk analysis."}

    if city_name not in df['city'].unique():
        return {"error": f"No data found for {city_name}."}

    # Filter city data
    city_data = df[df['city'] == city_name]

    # Calculate averages
    avg_temp = city_data['temperature'].mean()
    avg_humidity = city_data['humidity'].mean()
    avg_pressure = city_data['pressure'].mean()
    avg_wind = city_data['wind_speed'].mean()

    # 🌡️ Heat Risk
    if avg_temp > 35:
        heat_risk = "Extreme Heat Risk 🔥"
    elif avg_temp > 30:
        heat_risk = "High Heat Risk 🌡️"
    elif avg_temp > 25:
        heat_risk = "Moderate Heat Risk 🌤️"
    else:
        heat_risk = "Low Heat Risk ❄️"

    # 🌧️ Flood Risk (based on humidity and pressure)
    if avg_humidity > 85 and avg_pressure < 1005:
        flood_risk = "Severe Flood Risk 🌊"
    elif avg_humidity > 70:
        flood_risk = "Moderate Flood Risk 💧"
    else:
        flood_risk = "Low Flood Risk ☀️"

    # 💨 Discomfort Index (simplified Thom’s formula)
    discomfort_index = 0.5 * (avg_temp + 61.0 + ((avg_temp - 68.0) * 1.2) + (avg_humidity * 0.094))
    if discomfort_index > 80:
        comfort_level = "Very Uncomfortable 🥵"
    elif discomfort_index > 75:
        comfort_level = "Uncomfortable 😓"
    elif discomfort_index > 70:
        comfort_level = "Slightly Warm 🙂"
    else:
        comfort_level = "Comfortable 😌"

    # Combine all
    risk_summary = {
        "city": city_name,
        "avg_temp": round(avg_temp, 2),
        "avg_humidity": round(avg_humidity, 2),
        "avg_pressure": round(avg_pressure, 2),
        "avg_wind_speed": round(avg_wind, 2),
        "heat_risk": heat_risk,
        "flood_risk": flood_risk,
        "comfort_level": comfort_level,
    }

    return risk_summary
