# src/evaluate_models.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import predict_next_n_days
import joblib

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs("metrics", exist_ok=True)

def evaluate_city(city):
    """Evaluates RMSE, MAE, R2 for a single city's LSTM model."""
    data_path = os.path.join(DATA_DIR, f"{city.lower()}.csv")
    model_path = os.path.join(MODELS_DIR, f"lstm_{city.lower()}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{city.lower()}.pkl")

    if not (os.path.exists(data_path) and os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None

    try:
        df = pd.read_csv(data_path)
        if "temperature" not in df.columns:
            return None

        # Use last 14 days for evaluation
        actual = df["temperature"].tail(14).values
        forecast_df = predict_next_n_days(df, city, days=14)
        if forecast_df is None or forecast_df.empty:
            return None

        predicted = forecast_df["predicted_temperature"].values[: len(actual)]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return {"city": city, "RMSE": rmse, "MAE": mae, "R2": r2}
    except Exception:
        return None


def evaluate_all_cities():
    """Evaluates all available city models and saves metrics."""
    results = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            city = os.path.splitext(f)[0]
            print(f"Evaluating {city}...")
            metrics = evaluate_city(city)
            if metrics:
                results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("metrics/model_accuracy.csv", index=False)
        print("✅ Metrics saved to metrics/model_accuracy.csv")
    else:
        print("⚠️ No metrics computed — check data or models.")
    return df


if __name__ == "__main__":
    evaluate_all_cities()
