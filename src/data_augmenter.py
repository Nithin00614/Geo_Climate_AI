import pandas as pd
import numpy as np
from datetime import timedelta
from weather_fetcher import fetch_weather_data

def ensure_minimum_data(df, city, min_days=60):
    """
    Ensures the city has at least `min_days` of data.
    If not, fetches new data or simulates missing entries.
    """
    cdf = df[df["city"] == city].copy()
    cdf["date"] = pd.to_datetime(cdf["date"])
    cdf.sort_values("date", inplace=True)

    # Check data length
    if len(cdf) >= min_days:
        return cdf

    print(f"⚠️ Insufficient data for {city}: only {len(cdf)} days found. Augmenting...")

    # Try fetching latest API data
    try:
        new_data = fetch_weather_data(city)
        if new_data is not None and not new_data.empty:
            cdf = pd.concat([cdf, new_data]).drop_duplicates(subset=["date"], keep="last")
    except Exception as e:
        print(f"⚠️ Failed to fetch new data: {e}")

    # If still short, simulate
    while len(cdf) < min_days:
        last_row = cdf.iloc[-1]
        next_date = pd.to_datetime(last_row["date"]) + timedelta(days=1)
        temp_noise = np.random.normal(0, 0.5)
        hum_noise = np.random.normal(0, 1.0)
        rain_noise = np.random.normal(0, 0.2)
        new_entry = {
            "date": next_date,
            "city": city,
            "temperature": last_row["temperature"] + temp_noise,
            "humidity": min(100, max(0, last_row["humidity"] + hum_noise)),
            "rainfall": max(0, last_row["rainfall"] + rain_noise)
        }
        cdf = pd.concat([cdf, pd.DataFrame([new_entry])], ignore_index=True)

    print(f"✅ Data for {city} augmented to {len(cdf)} days.")
    return cdf
