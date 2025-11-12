# src/auto_retrain.py
import os
import pandas as pd
from datetime import datetime
from evaluate_models import evaluate_all_cities
from lstm_model import train_lstm_model
from weather_fetcher import fetch_weather_data

DATA_DIR = "data"
MODELS_DIR = "models"
METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

RMSE_THRESHOLD = 3.0
R2_THRESHOLD = 0.8

def identify_underperforming_models():
    """Identify cities whose models fail accuracy thresholds."""
    metrics_path = os.path.join(METRICS_DIR, "model_accuracy.csv")
    if not os.path.exists(metrics_path):
        print("⚠️ No metrics file found, running initial evaluation...")
        df = evaluate_all_cities()
    else:
        df = pd.read_csv(metrics_path)

    underperformers = df[
        (df["RMSE"] > RMSE_THRESHOLD) | (df["R2"] < R2_THRESHOLD)
    ]
    if underperformers.empty:
        print("✅ All models meet performance thresholds.")
    else:
        print(f"⚠️ {len(underperformers)} models need retraining.")
    return underperformers

def auto_retrain_models():
    """Retrain only the cities that need performance improvement."""
    underperformers = identify_underperforming_models()
    if underperformers.empty:
        return "No retraining required."

    retrain_log = []
    for _, row in underperformers.iterrows():
        city = row["city"]
        print(f"\n🔁 Retraining model for {city} (RMSE={row['RMSE']:.2f}, R²={row['R2']:.2f})...")

        data_path = os.path.join(DATA_DIR, f"{city.lower()}.csv")
        if not os.path.exists(data_path):
            print(f"🌤 Fetching new data for {city}...")
            df = fetch_weather_data(city)
            if df is None or df.empty:
                print(f"❌ Failed to fetch data for {city}.")
                continue
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)

        try:
            model, scaler = train_lstm_model(df, city)
            if model:
                print(f"✅ Retrained and saved model for {city}")
                retrain_log.append({
                    "city": city,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "old_RMSE": row["RMSE"],
                    "old_R2": row["R2"],
                    "status": "Success"
                })
            else:
                print(f"⚠️ Retraining failed for {city}")
        except Exception as e:
            print(f"❌ Error retraining {city}: {e}")

    if retrain_log:
        log_df = pd.DataFrame(retrain_log)
        log_path = os.path.join(METRICS_DIR, "retrain_log.csv")
        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        log_df.to_csv(log_path, index=False)
        print(f"\n🧾 Retraining summary saved to {log_path}")
    else:
        print("⚠️ No successful retraining logged.")

    print("\n🔁 Re-evaluating all models...")
    evaluate_all_cities()

if __name__ == "__main__":
    auto_retrain_models()
