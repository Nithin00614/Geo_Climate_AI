import os
import requests
import pandas as pd
from pathlib import Path
from time import sleep

OUT_DIR = Path("data/real_weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
PAUSE = 1.5  # seconds (avoid rate limit)

CITY_COORDS = {
    # --- India Major Cities ---
    "bengaluru": (12.9716, 77.5946),
    "delhi": (28.7041, 77.1025),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "nagpur": (21.1458, 79.0882),
    "visakhapatnam": (17.6868, 83.2185),
    "bhopal": (23.2599, 77.4126),
    "patna": (25.5941, 85.1376),
    "ranchi": (23.3441, 85.3096),
    "coimbatore": (11.0168, 76.9558),
    "kochi": (9.9312, 76.2673),
    "trivandrum": (8.5241, 76.9366),
    "mysuru": (12.2958, 76.6394),
    "mangaluru": (12.9141, 74.8560),
    "bhubaneswar": (20.2961, 85.8245),
    "surat": (21.1702, 72.8311),
    "indore": (22.7196, 75.8577),
    "vadodara": (22.3072, 73.1812),
    "rajkot": (22.3039, 70.8022),
    "meerut": (28.9845, 77.7064),
    "jodhpur": (26.2389, 73.0243),
    "guwahati": (26.1445, 91.7362),

    # --- Global Cities ---
    "new_york": (40.7128, -74.0060),
    "los_angeles": (34.0522, -118.2437),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "dubai": (25.2048, 55.2708),
    "tokyo": (35.6762, 139.6503),
    "osaka": (34.6937, 135.5023),
    "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "toronto": (43.6532, -79.3832),
    "vancouver": (49.2827, -123.1207),
    "san_francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "boston": (42.3601, -71.0589),
    "chicago": (41.8781, -87.6298),
    "mexico_city": (19.4326, -99.1332),
    "rio": (-22.9068, -43.1729),
    "cairo": (30.0444, 31.2357),
    "istanbul": (41.0082, 28.9784),
    "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964),
    "moscow": (55.7558, 37.6176),
    "seoul": (37.5665, 126.9780),
    "jakarta": (-6.2088, 106.8456),
    "manila": (14.5995, 120.9842),
    "bangkok": (13.7563, 100.5018),
    "kathmandu": (27.7172, 85.3240),
}


START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"


def safe_fetch(url):
    """Fetch with retry and graceful fallback."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=50)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        print(f"Retry {attempt}/{MAX_RETRIES} ...")
        sleep(PAUSE)

    return None


def fetch_city(city, lat, lon):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        "precipitation_sum,wind_speed_10m_max"
        "&timezone=UTC"
    )

    data = safe_fetch(url)
    if not data or "daily" not in data:
        print(f"⚠️  No data for {city}. Skipping.")
        return None

    d = data["daily"]
    df = pd.DataFrame({
        "date": d["time"],
        "temperature": d["temperature_2m_mean"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "rainfall": d["precipitation_sum"],
        "wind_speed": d["wind_speed_10m_max"],
        "city": city
    })

    return df


def main():
    for city, (lat, lon) in CITY_COORDS.items():
        print(f"\n🔄 Fetching: {city}...")
        df = fetch_city(city, lat, lon)
        if df is None:
            continue
        path = OUT_DIR / f"{city}.csv"
        df.to_csv(path, index=False)
        print(f"✅ Saved {city}: {len(df)} rows")
        sleep(PAUSE)  # avoid rate limit


if __name__ == "__main__":
    main()
