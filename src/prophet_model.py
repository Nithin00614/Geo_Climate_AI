from prophet import Prophet
import pandas as pd
import os

def train_prophet(city):
    df = pd.read_csv("data/historical_combined.csv")
    df = df[df["city"] == city][["date", "temp_max"]]
    df.rename(columns={"date": "ds", "temp_max": "y"}, inplace=True)

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)
    forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(f"data/forecast_{city.lower()}.csv", index=False)
    print(f"✅ Prophet forecast generated for {city}")

if __name__ == "__main__":
    for c in ["Chennai","Bengaluru","Delhi"]:
        train_prophet(c)
