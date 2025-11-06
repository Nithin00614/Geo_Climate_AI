import numpy as np
import pandas as pd

def analyze_climate_risk(df, city_name, predicted_temps):
    """
    Analyze climate risks (heatwave, drought, flood potential)
    based on temperature trends and humidity.
    """

    city_df = df[df["city"].str.lower() == city_name.lower()]
    if city_df.empty:
        print(f"⚠️ No data found for {city_name}")
        return

    avg_temp = city_df["temperature"].mean()
    avg_humidity = city_df["humidity"].mean()

    future_avg = np.mean(predicted_temps)

    print("\n🌡️ --- Climate Risk Analysis ---")
    print(f"📍 City: {city_name}")
    print(f"🕐 Past Avg Temp: {avg_temp:.2f} °C")
    print(f"🕐 Predicted Avg Temp (Next 7 Days): {future_avg:.2f} °C")
    print(f"💧 Avg Humidity: {avg_humidity:.2f}%")

    risk = "Low"
    risk_type = "Normal Conditions"

    if future_avg > avg_temp + 5 and avg_humidity < 50:
        risk = "High"
        risk_type = "🔥 Heatwave / Drought Risk"
    elif future_avg < avg_temp - 5 and avg_humidity > 70:
        risk = "Moderate"
        risk_type = "🌊 Possible Flood / High Rain Risk"
    elif abs(future_avg - avg_temp) < 2:
        risk = "Low"
        risk_type = "🌤️ Stable Climate Trend"

    print(f"\n📊 Risk Level: {risk}")
    print(f"⚠️ Risk Type: {risk_type}")
    print("----------------------------")

    return {
        "city": city_name,
        "past_avg_temp": avg_temp,
        "future_avg_temp": future_avg,
        "avg_humidity": avg_humidity,
        "risk_level": risk,
        "risk_type": risk_type
    }
