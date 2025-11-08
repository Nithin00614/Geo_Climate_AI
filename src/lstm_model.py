import os
import numpy as np
import pandas as pd
import joblib
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# -------------------------------
# CONFIG
# -------------------------------
TIME_STEPS = 30
FEATURES = ["temp_max", "temp_min", "rainfall"]


# -------------------------------
# TRAINING FUNCTION (MULTI-FEATURE LSTM)
# -------------------------------
def train_multifeature_lstm(city, force_retrain=False):
    """
    Trains an LSTM model for a specific city using multiple weather features.
    If model already exists, skips training unless force_retrain=True.
    """
    model_path = f"models/lstm_{city.lower()}.keras"
    scaler_path = f"models/scaler_{city.lower()}.pkl"

    # ✅ Skip retraining if already exists (unless forced)
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"⚙️ Model for {city} already exists — skipping retrain.")
        return None, None

    data_path = "data/historical_combined.csv"
    if not os.path.exists(data_path):
        print("❌ Dataset not found. Please run the data pipeline first.")
        return None, None

    df = pd.read_csv(data_path)
    df = df[df["city"] == city].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    if len(df) < TIME_STEPS + 1:
        print(f"⚠️ Not enough data to train model for {city}.")
        return None, None

    data = df[FEATURES]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(TIME_STEPS, len(scaled_data)):
        X.append(scaled_data[i - TIME_STEPS:i])
        y.append(scaled_data[i, 0])  # target = temp_max

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=25, batch_size=16, verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Trained and saved model for {city}")
    return model, scaler


# -------------------------------
# FORECAST FUNCTION
# -------------------------------
def predict_next_7_days(df, city, scaler=None):
    """
    Predicts the next 7 days of temperature for a given city using trained LSTM.
    """
    model_path = f"models/lstm_{city.lower()}.keras"
    scaler_path = f"models/scaler_{city.lower()}.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"⚠️ Model or scaler missing for {city}. Train the model first.")
        return None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    city_df = df[df["city"] == city].copy()
    city_df["date"] = pd.to_datetime(city_df["date"])
    city_df.sort_values("date", inplace=True)

    scaled = scaler.transform(city_df[FEATURES])
    last_seq = scaled[-TIME_STEPS:]

    predictions = []
    for _ in range(7):
        pred = model.predict(np.expand_dims(last_seq, axis=0), verbose=0)[0][0]
        # simulate next input by shifting window
        new_row = np.append(last_seq[-1, 1:], pred)
        last_seq = np.vstack([last_seq[1:], new_row])
        predictions.append(pred)

    dummy = np.zeros((len(predictions), len(FEATURES)))
    dummy[:, 0] = predictions
    forecast_temp = scaler.inverse_transform(dummy)[:, 0]

    future_dates = pd.date_range(city_df["date"].max() + timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_temperature": forecast_temp
    })
    return forecast_df


# -------------------------------
# AUTO TRAIN FOR ALL CITIES
# -------------------------------
def auto_train_all_cities(df):
    """
    Trains LSTM models for all cities that don't already have a saved model.
    """
    if df is None or df.empty:
        print("❌ No data provided for auto-training.")
        return

    cities = sorted(df["city"].unique())
    for city in cities:
        try:
            model_path = f"models/lstm_{city.lower()}.keras"
            scaler_path = f"models/scaler_{city.lower()}.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print(f"⚙️ Skipping {city} — model already exists.")
                continue

            print(f"🔄 Training model for {city}...")
            train_multifeature_lstm(city)
        except Exception as e:
            print(f"❌ Error training {city}: {e}")

    print("✅ Auto-training complete for all missing cities!")


# -------------------------------
# MANUAL TEST ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    data_path = "data/historical_combined.csv"
    if not os.path.exists(data_path):
        print("❌ Dataset not found. Please run the data pipeline first.")
    else:
        df = pd.read_csv(data_path)
        sample_city = df["city"].unique()[0]
        print(f"🧠 Testing LSTM model for {sample_city}")
        train_multifeature_lstm(sample_city, force_retrain=True)
        forecast = predict_next_7_days(df, sample_city)
        print(forecast)
