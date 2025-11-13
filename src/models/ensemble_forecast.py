import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# --- Import your models ---
from models.lstm_model import load_lstm_model
from models.xgb_model import load_xgb_model
from models.prophet_model import run_prophet_forecast


TIME_STEPS = 14  # Window size

# -----------------------------------------------------
# Helper function: safe scaler
# -----------------------------------------------------
def prepare_scaler(df):
    scaler = MinMaxScaler()
    try:
        scaled = scaler.fit_transform(df)
    except:
        print("⚠️ Could not scale with existing scaler. Re-fitting.")
        df = df.fillna(0)
        scaled = scaler.fit_transform(df)
    return scaler, scaled


# -----------------------------------------------------
# LSTM FORECAST
# -----------------------------------------------------
def lstm_forecast(df, city, model, scaler):
    try:
        feature_cols = ["temperature", "humidity", "rainfall", "wind_speed"]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        feats = df[feature_cols]
        scaled = scaler.transform(feats)

        window = scaled[-TIME_STEPS:]
        if window.shape[0] < TIME_STEPS:
            print(f"⚠️ Not enough data for LSTM for {city}. Filling window.")
            padding = np.zeros((TIME_STEPS - window.shape[0], window.shape[1]))
            window = np.vstack([padding, window])

        preds = []
        current = window.copy()

        for _ in range(14):
            p = model.predict(current.reshape(1, TIME_STEPS, len(feature_cols)))
            preds.append(float(p[0]))

            next_step = np.zeros((1, len(feature_cols)))
            next_step[0, 0] = p  # only temperature predicted

            current = np.vstack([current[1:], next_step])

        return preds

    except Exception as e:
        print(f"❌ LSTM forecast error for {city}: {e}")
        return [np.nan] * 14


# -----------------------------------------------------
# XGBOOST FORECAST
# -----------------------------------------------------
def xgb_forecast(df, city, model):
    try:
        feature_cols = ["temperature", "humidity", "rainfall", "wind_speed"]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        feats = df[feature_cols].tail(1).values
        preds = []

        last_vals = feats.copy()
        for _ in range(14):
            p = model.predict(last_vals)[0]
            preds.append(float(p))

            last_vals = np.array([[p, last_vals[0][1], last_vals[0][2], last_vals[0][3]]])

        return preds

    except Exception as e:
        print(f"❌ XGBoost forecast error for {city}: {e}")
        return [np.nan] * 14


# -----------------------------------------------------
# MAIN ENSEMBLE FORECAST
# -----------------------------------------------------
def ensemble_forecast(df, city):

    df = df.sort_values("date")

    # Load models
    lstm_model, scaler = load_lstm_model(city)
    xgb_model = load_xgb_model(city)

    # LSTM predictions
    lstm_pred = lstm_forecast(df.copy(), city, lstm_model, scaler)

    # Prophet predictions
    prophet_pred = run_prophet_forecast(df.copy(), horizon=14)

    # XGBoost predictions
    xgb_pred = xgb_forecast(df.copy(), city, xgb_model)

    # Clean arrays
    y_lstm = np.array(lstm_pred, dtype=float)
    y_prophet = np.array(prophet_pred, dtype=float)
    y_xgb = np.array(xgb_pred, dtype=float)

    # Weight based on recent performance (placeholder static weights)
    w_lstm = 0.4
    w_prophet = 0.3
    w_xgb = 0.3

    final = (y_lstm * w_lstm) + (y_prophet * w_prophet) + (y_xgb * w_xgb)

    # Dates for next 14 days
    start_date = df["date"].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(14)]

    # Output dataframe
    out = pd.DataFrame({
        "date": future_dates,
        "yhat_lstm": y_lstm,
        "yhat_prophet": y_prophet,
        "yhat_xgb": y_xgb,
        "predicted_temperature": final,
        "lstm_weight": [w_lstm] * 14,
        "prophet_weight": [w_prophet] * 14,
        "xgb_weight": [w_xgb] * 14
    })

    return out
