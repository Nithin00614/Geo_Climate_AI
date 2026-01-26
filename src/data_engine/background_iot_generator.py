import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
import time
from datetime import datetime
from src.utils.db import init_db, insert_iot_row
from src.data_engine.risk_engine import compute_risk
from src.data_engine.email_engine import send_email_alert

CITIES = {
    "bengaluru": {"base_temp": 28, "base_hum": 55, "base_aqi": 80},
    "mumbai": {"base_temp": 30, "base_hum": 70, "base_aqi": 90},
    "delhi": {"base_temp": 32, "base_hum": 40, "base_aqi": 150},
    "chennai": {"base_temp": 33, "base_hum": 60, "base_aqi": 70},
    "kolkata": {"base_temp": 31, "base_hum": 65, "base_aqi": 110},
}

INTERVAL = 3

def generate_sensor_row(city):
    base = CITIES[city]

    temp = round(base["base_temp"] + random.uniform(-3, 4), 2)
    hum = round(base["base_hum"] + random.uniform(-10, 10), 2)
    pres = round(1010 + random.uniform(-15, 15), 2)
    wind = round(random.uniform(0.5, 8), 2)
    aqi = round(base["base_aqi"] + random.uniform(-30, 40), 2)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": city,
        "temperature": temp,
        "humidity": hum,
        "pressure": pres,
        "wind_speed": wind,
        "aqi": aqi
    }

def run_generator():
    init_db()
    print("\n🚀 IoT Climate Engine Started...\n")

    while True:
        for city in CITIES.keys():
            row = generate_sensor_row(city)

            risk, severity = compute_risk(row)
            row["risk"] = round(risk, 2)
            row["severity"] = severity

            insert_iot_row(row)

            print(f"📥 {city} -> {row}")

            if severity == "High":
                send_email_alert(
                    subject=f"🚨 High Climate Alert in {city}",
                    body=f"""
High climate risk detected!
City: {city}
Temperature: {row['temperature']}
Humidity: {row['humidity']}
AQI: {row['aqi']}
Risk Score: {row['risk']}
Severity: {severity}
Time: {row['timestamp']}
"""
                )

        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_generator()
