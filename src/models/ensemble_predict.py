# src/models/ensemble_predict.py
import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

MODEL_DIR = "src/models"

def load_xgb(city):
    fn = os.path.join(MODEL_DIR, f"xgb_{city}.joblib")
    if os.path.exists(fn):
        model, cols = joblib.load(fn)
        return model, cols
    return None, None

def load_prophet(city):
    fn = os.path.join(MODEL_DIR, f"prophet_{city}.pkl")
    if os.path.exists(fn):
        m = joblib.load(fn)
        return m
    return None

def load_lstm(city):
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    fname = os.path.join(MODEL_DIR, f"lstm_{city}.h5")
    scaler_fn = os.path.join(MODEL_DIR, f"lstm_scaler_{city}.joblib")
    if os.path.exists(fname) and os.path.exists(scaler_fn):
        model = load_model(fname)
        scaler = joblib.load(scaler_fn)
        return model, scaler
    return None, None

def predict_xgb(model_cols, model, df, horizon=14):
    # df must include the most recent rows to compute lags and rolling
    df_feat = df.copy()
    for lag in [1,2,3,7,14]:
        df_feat[f"temp_lag_{lag}"] = df_feat["temperature"].shift(lag)
    df_feat["temp_roll_7"] = df_feat["temperature"].rolling(7, min_periods=1).mean()
    df_feat["temp_roll_14"] = df_feat["temperature"].rolling(14, min_periods=1).mean()
    # take last row to create features for day1, then iteratively update using predictions
    last = df_feat.iloc[-1:].copy()
    preds = []
    for i in range(horizon):
        Xrow = last[model_cols].fillna(method="ffill").fillna(method="bfill").values
        yhat = model.predict(pd.DataFrame([Xrow], columns=model_cols))[0]
        preds.append(yhat)
        # roll forward: drop oldest lag and insert new yhat
        # create a new row for next iteration
        new_row = last.copy()
        # update temperature column to predicted for future lags
        new_row["temperature"] = yhat
        # recompute lag features for next step
        for lag in [1,2,3,7,14]:
            # shift by 1 day: temp_lag_1 becomes predicted temp, temp_lag_2 becomes prev temp_lag_1 etc.
            col = f"temp_lag_{lag}"
            if lag == 1:
                new_row[col] = yhat
            else:
                prev = last[f"temp_lag_{lag-1}"].values[0] if f"temp_lag_{lag-1}" in last.columns else last["temperature"].values[0]
                new_row[col] = prev
        new_row["temp_roll_7"] = (last["temp_roll_7"].values[0] * 6 + yhat) / 7
        new_row["temp_roll_14"] = (last["temp_roll_14"].values[0] * 13 + yhat) / 14
        last = new_row
    return np.array(preds)

def predict_prophet(model, df, horizon=14):
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)
    # take last horizon rows
    out = forecast[['ds','yhat']].tail(horizon).rename(columns={'ds':'date','yhat':'yhat_prophet'})
    out['date'] = pd.to_datetime(out['date'])
    return out

def predict_lstm(lstm_model, scaler, df, horizon=14, seq_len=30):
    # use multivariate inputs scaled by scaler
    features = ["temperature","humidity","pressure","wind_speed"]
    data = df[features].fillna(method="ffill").fillna(method="bfill").values
    scaled = scaler.transform(data)
    seq = scaled[-seq_len:]
    preds = []
    cur_seq = seq.copy()
    for _ in range(horizon):
        inp = cur_seq.reshape((1, cur_seq.shape[0], cur_seq.shape[1]))
        yhat = lstm_model.predict(inp, verbose=0)[0,0]
        preds.append(yhat)
        # append new row: build fake full feature row by copying last and replacing temperature scaled
        next_row = cur_seq[-1].copy()
        next_row[0] = yhat
        cur_seq = np.vstack([cur_seq[1:], next_row])
    # preds are in scaled temperature space? If we trained to scale multivariate and used scaler on full features,
    # we saved temperature scaler as part of scaler; we need to invert transform. For simplicity, we'll assume temperature scaled
    # is at index 0 and scaler has min_/scale_ for each feature.
    # Inverse transform each predicted temperature (only temperature):
    temp_min = scaler.data_min_[0]
    temp_scale = scaler.data_range_[0]
    inv_preds = [p * temp_scale + temp_min for p in preds]
    return np.array(inv_preds)

def run_ensemble_forecast(df_recent, city, horizon=14):
    """
    df_recent: pandas dataframe with recent full history for the city (columns include date, temperature, humidity,..)
    returns: DataFrame with date and columns yhat_lstm, yhat_prophet, yhat_xgb, final_pred
    """
    xgb_model, xgb_cols = None, None
    m_prophet = None
    lstm_model, lstm_scaler = None, None

    try:
        xgb_model, xgb_cols = load_xgb(city)
    except Exception:
        xgb_model, xgb_cols = None, None
    try:
        m_prophet = load_prophet(city)
    except Exception:
        m_prophet = None
    try:
        lstm_model, lstm_scaler = load_lstm(city)
    except Exception:
        lstm_model, lstm_scaler = None, None

    # Prepare dates
    last_date = pd.to_datetime(df_recent['date']).max()
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

    # Predictions default None
    preds = {
        "date": dates,
        "yhat_lstm": [None]*horizon,
        "yhat_prophet": [None]*horizon,
        "yhat_xgb": [None]*horizon
    }

    if xgb_model and xgb_cols:
        try:
            p = predict_xgb(xgb_cols, xgb_model, df_recent, horizon=horizon)
            preds["yhat_xgb"] = p.tolist()
        except Exception as e:
            print("XGB predict failed:", e)
    if m_prophet:
        try:
            p = predict_prophet(m_prophet, df_recent, horizon=horizon)
            # align by date
            pp = pd.Series(p["yhat_prophet"].values, index=dates)
            preds["yhat_prophet"] = pp.values.tolist()
        except Exception as e:
            print("Prophet predict failed:", e)
    if lstm_model and lstm_scaler:
        try:
            p = predict_lstm(lstm_model, lstm_scaler, df_recent, horizon=horizon)
            preds["yhat_lstm"] = p.tolist()
        except Exception as e:
            print("LSTM predict failed:", e)

    out = pd.DataFrame(preds)
    # combine final_pred as weighted mean of available preds (equal weights on available)
    def row_comb(r):
        vals = [v for v in [r["yhat_lstm"], r["yhat_prophet"], r["yhat_xgb"]] if v is not None]
        if not vals:
            return None
        return float(np.mean(vals))
    out["final_pred"] = out.apply(row_comb, axis=1)
    return out
