# src/models/lstm_model.py
"""
lstm_model.py
Provides:
- train_lstm_model(df, city, save_dir='models/lstm', epochs=20, time_steps=7)
- lstm_predict_next(df, horizon, city=None, save_dir='models/lstm')
- load_lstm_model(city, save_dir='models/lstm')
If TensorFlow is not installed, functions will raise informative errors for training and
predict will be left to forecast_engine fallback.
"""

import os
import joblib
import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
except Exception:
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _get_series(df):
    for col in ["temperature","temp","y","value"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').dropna().reset_index(drop=True)
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        return numeric.iloc[:, -1].dropna().reset_index(drop=True)
    return pd.Series([])

def _create_sequences(values, time_steps=7):
    x, y = [], []
    for i in range(len(values)-time_steps):
        x.append(values[i:i+time_steps])
        y.append(values[i+time_steps])
    if not x:
        return np.array([]), np.array([])
    return np.array(x), np.array(y)

def train_lstm_model(df, city, save_dir="models/lstm", epochs=30, time_steps=7, verbose=0):
    """
    Train per-city LSTM. Expects df with a numeric target column (temperature).
    Saves model and scaler. Returns (model_path, scaler_path).
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available in this environment. Install tensorflow to train LSTM.")

    _ensure_dir(save_dir)
    s = _get_series(df)
    if s.empty or len(s) < time_steps + 2:
        raise ValueError("Not enough data to train LSTM; need at least time_steps+2 rows.")

    scaler = MinMaxScaler()
    vals = scaler.fit_transform(s.values.reshape(-1,1)).flatten()

    X, Y = _create_sequences(vals, time_steps=time_steps)
    if X.size == 0:
        raise ValueError("Failed to create LSTM sequences (too little data).")

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, input_shape=(time_steps,1), return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0)
    model.fit(X, Y, epochs=epochs, batch_size=8, callbacks=[es], verbose=verbose)

    model_path = os.path.join(save_dir, f"{city}.keras")
    scaler_path = os.path.join(save_dir, f"{city}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path

def load_lstm_model(city, save_dir="models/lstm"):
    model_path = os.path.join(save_dir, f"{city}.keras")
    scaler_path = os.path.join(save_dir, f"{city}_scaler.pkl")
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Saved LSTM model or scaler not found.")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def lstm_predict_next(df, horizon=14, city=None, save_dir="models/lstm", time_steps=7):
    """
    Predict next horizon days using saved model for the city.
    If model/scaler missing or TF unavailable, raises to let forecast_engine fallback.
    Returns a numpy array length=horizon.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available for LSTM predict.")
    if city is None:
        raise ValueError("city argument required for LSTM predict (to find saved model).")
    model_path = os.path.join(save_dir, f"{city}.keras")
    scaler_path = os.path.join(save_dir, f"{city}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("LSTM model or scaler not found.")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    s = _get_series(df)
    if s.empty:
        raise ValueError("No numeric series available for LSTM prediction.")
    scaled = scaler.transform(s.values.reshape(-1,1)).flatten()
    
    # Calculate residual std for stochastic variation
    raw_values = s.values
    if len(raw_values) > 2:
        residual_std = np.std(np.diff(raw_values)) * 0.5
    else:
        residual_std = 0.1
    
    # create rolling input of last time_steps
    seq = list(scaled[-time_steps:])
    preds = []
    for _ in range(horizon):
        x_in = np.array(seq[-time_steps:]).reshape(1, time_steps, 1)
        p = model.predict(x_in, verbose=0)[0][0]
        # Add stochastic noise for variation
        noise = np.random.normal(0, residual_std / (np.max(raw_values) - np.min(raw_values) + 1e-6))
        p = np.clip(p + noise * 0.01, 0, 1)  # Keep within scaled bounds
        pred_val = float(scaler.inverse_transform([[p]])[0][0])
        preds.append(pred_val)
        seq.append(p)
    return np.array(preds, dtype=float)
