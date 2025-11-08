# src/iot_simulator.py
import random
import pandas as pd
from datetime import datetime

# Basic lat/lon lookup for your cities (extend if needed)
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Chandigarh": (30.7333, 76.7794),
    "Surat": (21.1702, 72.8311),
    "Madurai": (9.9252, 78.1198)
}

def generate_mock_iot_data(city):
    """
    Returns a dict with mock sensor readings:
    - temperature (°C)
    - humidity (%)
    - rainfall (mm)
    - timestamp (pd.Timestamp)
    - lat, lon
    """
    # sensible ranges; you can tweak these
    temp = round(random.uniform(18.0, 42.0), 2)
    humidity = round(random.uniform(30.0, 95.0), 2)
    rain = round(max(0, random.gauss(3.0, 8.0)), 2)  # skew toward small rainfall but occasionally large
    timestamp = pd.Timestamp.now()

    lat, lon = CITY_COORDS.get(city, (0.0, 0.0))
    return {
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rain,
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon
    }

def batch_generate(cities):
    """
    Generate one reading per city (list of dicts).
    """
    rows = []
    for c in cities:
        rows.append(generate_mock_iot_data(c))
    return rows
