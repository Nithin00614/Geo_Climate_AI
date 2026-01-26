import pandas as pd
from datetime import timedelta

def run_ensemble_forecast(df, forecast_days=14):
    """Simple fallback ensemble: trend-line forecast if ML models missing."""

    df = df.sort_values("date")
    last_val = df["temperature"].iloc[-1]

    # Simple linear slope using last 7 days
    if len(df) >= 7:
        slope = (df["temperature"].iloc[-1] - df["temperature"].iloc[-7]) / 7
    else:
        slope = 0.1

    forecast_dates = [df["date"].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast_vals = [(last_val + slope * i) for i in range(1, forecast_days + 1)]

    forecast_df = pd.DataFrame({"date": forecast_dates, "forecast": forecast_vals})

    weights = {
        "lstm": 0,      # Placeholder until ML models are restored
        "prophet": 0,
        "xgb": 0
    }

    return forecast_df, weights
