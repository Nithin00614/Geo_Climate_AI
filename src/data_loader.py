import pandas as pd
import os

def load_weather_data(file_path='data/weather.csv'):
    """
    Loads weather data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Weather data not found at {file_path}. "
            "Place a valid weather.csv file inside the 'data' folder."
        )
    df = pd.read_csv(file_path)
    return df
