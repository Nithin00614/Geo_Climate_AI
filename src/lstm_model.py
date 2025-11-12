# src/lstm_model.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Config
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["temperature", "humidity", "rainfall"]
TIME_STEPS = 30  # LSTM input window
DEFAULT_EPOCHS = 12
DEFAULT_BATCH = 16

def _normalize_df(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    # rename temp -> temperature if present
    if "temp" in df.columns and "temperature" not in df.columns:
        df.rename(columns={"temp": "temperature"}, inplace=True)

    # fill missing columns with sensible defaults
    n = len(df)
    if "temperature" not in df.columns:
        df["temperature"] = np.random.uniform(22, 30, n)
    if "humidity" not in df.columns:
        df["humidity"] = np.clip(np.random.normal(60, 10, n), 10, 100)
    if "rainfall" not in df.columns:
        df["rainfall"] = np.zeros(n)

    # date handling
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df

def create_sequences(data, time_steps=TIME_STEPS):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # predict temperature (first column)
    return np.array(X), np.array(y)

def train_lstm_model(df, city, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH):
    """
    Train and save an LSTM for `city`. Returns (model, scaler) or (None, None).
    """
    try:
        city_df = df[df["city"].astype(str).str.lower() == city.lower()].copy()
        if city_df.empty:
            return None, None

        city_df = _normalize_df(city_df)
        values = city_df[FEATURES].values.astype(float)

        # pad if too short so we can form sequences
        if len(values) < TIME_STEPS + 1:
            pad_needed = (TIME_STEPS + 1) - len(values)
            pad_row = values[-1].reshape(1, -1)
            pad = np.repeat(pad_row, pad_needed, axis=0)
            values = np.vstack([values, pad])

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        X, y = create_sequences(scaled, time_steps=TIME_STEPS)
        if len(X) == 0:
            return None, None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        model_path = os.path.join(MODEL_DIR, f"lstm_{city.lower()}.keras")
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{city.lower()}.pkl")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        return model, scaler
    except Exception as e:
        print("train_lstm_model error:", e)
        return None, None

def auto_train_all_cities(df):
    cities = sorted(df["city"].astype(str).unique())
    for c in cities:
        print("Training:", c)
        train_lstm_model(df, c)

def predict_next_n_days(df, city, days=14):
    """
    Load saved LSTM + scaler and predict next `days` temperatures.
    Returns DataFrame: ['date','predicted_temperature'] or None.
    """
    try:
        city = str(city)
        model_path = os.path.join(MODEL_DIR, f"lstm_{city.lower()}.keras")
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{city.lower()}.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            # model missing
            return None

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        city_df = df[df["city"].astype(str).str.lower() == city.lower()].copy()
        if city_df.empty:
            return None

        city_df = _normalize_df(city_df)
        values = city_df[FEATURES].values.astype(float)

        # pad to at least TIME_STEPS
        if values.shape[0] < TIME_STEPS:
            pad_needed = TIME_STEPS - values.shape[0]
            pad_row = values[-1].reshape(1, -1)
            pad = np.repeat(pad_row, pad_needed, axis=0)
            values = np.vstack([values, pad])

        scaled = scaler.transform(values)
        X_input = np.array([scaled[-TIME_STEPS:, :]])  # shape (1, TIME_STEPS, n_features)

        preds = []
        for _ in range(days):
            out = model.predict(X_input, verbose=0)
            p = float(out.flatten()[0])
            preds.append(p)
            # next row: predicted temp + reuse last humidity/rainfall
            last_h = X_input[0, -1, 1] if X_input.shape[2] > 1 else 0
            last_r = X_input[0, -1, 2] if X_input.shape[2] > 2 else 0
            next_row = np.array([p, last_h, last_r]).reshape(1, 1, -1)
            X_input = np.concatenate([X_input[:, 1:, :], next_row], axis=1)

        preds = np.array(preds).reshape(-1, 1)
        zeros = np.zeros((preds.shape[0], 2))
        stacked = np.hstack([preds, zeros])
        inv = scaler.inverse_transform(stacked)[:, 0]

        last_date = pd.to_datetime(city_df["date"].iloc[-1])
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)
        return pd.DataFrame({"date": future_dates, "predicted_temperature": inv})
    except Exception as e:
        print("predict_next_n_days error:", e)
        return None
