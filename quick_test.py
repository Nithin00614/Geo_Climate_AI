# quick_test.py (place anywhere)
import pandas as pd
from src.models.ensemble_predict import run_ensemble_forecast

df = pd.read_csv("data/real_weather/ahmedabad.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
out = run_ensemble_forecast(df, "ahmedabad", horizon=14)
print(out.head())
