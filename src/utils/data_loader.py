import os
import pandas as pd

DATA_FOLDER = "src/data/real_weather"

def load_all_city_data():
    cities = {}

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            path = os.path.join(DATA_FOLDER, file)

            try:
                df = pd.read_csv(path)

                # Standardize column names
                df.columns = [c.lower().strip() for c in df.columns]

                # Fix date parsing
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # Rename temperature column if needed
                if "temp" in df.columns:
                    df.rename(columns={"temp": "temperature"}, inplace=True)

                city = file.replace(".csv", "").replace("_", " ").title()

                df = df.dropna(subset=["date", "temperature"])
                df = df.sort_values("date")

                cities[city] = df

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    return cities


def load_city_data(city: str):
    all_data = load_all_city_data()
    return all_data.get(city, None)
