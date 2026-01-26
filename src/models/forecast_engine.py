# src/models/forecast_engine.py
"""
forecast_engine.py
Unified ensemble wrapper that attempts to call real models (if present) and falls back safely.

Public:
- ensemble_forecast(history_df, city=None, horizon=14)
- run_full_ensemble_forecast(history_df, city=None, horizon=14)  (backwards compatible)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import importlib
import traceback

# helpers to call model predict functions dynamically
def _call_if_available(module_name, func_name, *args, **kwargs):
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ {_short_exc(e)} calling {module_name}.{func_name}")
    raise ImportError(f"{module_name}.{func_name} not available")

def _short_exc(e):
    try:
        return str(e)
    except:
        return repr(e)

def _get_date_col(df):
    for c in df.columns:
        if 'date' in c.lower() or c.lower() in ('ds','timestamp','time'):
            return c
    return None

def _persistence(df, horizon):
    """Persistence forecast with seasonal component."""
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if s.empty:
                continue
            last = s.values
            if last.size < 7:
                return np.array([float(last[-1])] * horizon, dtype=float)
            # Add weekly seasonality
            weekly_pattern = s.values[-7:]  # Last 7 days
            preds = []
            for i in range(horizon):
                seasonal_idx = i % 7
                preds.append(float(weekly_pattern[seasonal_idx]))
            return np.array(preds, dtype=float)
    return np.array([np.nan]*horizon, dtype=float)

def _linear_trend(df, horizon):
    """Linear trend with seasonal variation."""
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) < 2:
                return _persistence(df, horizon)
            x = np.arange(len(s))
            a, b = np.polyfit(x, s.values, 1)
            future_x = np.arange(len(s), len(s)+horizon)
            trend = a*future_x + b
            
            # Add seasonal component
            if len(s) >= 7:
                detrended = s.values - (a * x + b)
                seasonal = detrended[-7:]
                preds = []
                for i, t_val in enumerate(trend):
                    seasonal_idx = i % 7
                    preds.append(float(t_val + seasonal[seasonal_idx]))
                return np.array(preds, dtype=float)
            return trend.astype(float)
    return np.array([np.nan]*horizon, dtype=float)

def _moving_average(df, horizon, window=5):
    """Moving average with random walk variation."""
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if s.empty:
                return _persistence(df, horizon)
            
            base_mean = float(s.tail(window).mean())
            std_dev = float(s.std()) if len(s) > 1 else 1.0
            
            # Create predictions with slight random variation
            preds = []
            current = base_mean
            for _ in range(horizon):
                # Small random walk with mean reversion
                noise = np.random.normal(0, std_dev * 0.3)
                current = base_mean * 0.1 + current * 0.9 + noise
                preds.append(float(current))
            return np.array(preds, dtype=float)
    return np.array([np.nan]*horizon, dtype=float)

def _safe_call_predictors(df, city, horizon):
    """
    Tries to call:
     - src.models.lstm_model.lstm_predict_next(df, horizon, city=city)
     - src.models.prophet_model.prophet_forecast(df, horizon, city=city)
     - src.models.xgb_model.xgb_predict(df, horizon, city=city)
    If any call fails (ImportError/FileMissing/Runtime), fallback to the simple baseline.
    Returns tuple of numpy arrays (y_lstm,y_prophet,y_xgb)
    """
    # LSTM
    y_lstm = None
    try:
        y_lstm = _call_if_available("src.models.lstm_model", "lstm_predict_next", df, horizon, city=city)
        y_lstm = np.asarray(y_lstm, dtype=float)
    except Exception:
        try:
            y_lstm = _call_if_available("src.models.lstm_model", "lstm_predict_next", df, horizon, city)
        except Exception:
            print("ℹ️ LSTM unavailable or failed — using persistence fallback.")
            y_lstm = _persistence(df, horizon)

    # Prophet
    y_prophet = None
    try:
        y_prophet = _call_if_available("src.models.prophet_model", "prophet_forecast", df, horizon, city=city)
        y_prophet = np.asarray(y_prophet, dtype=float)
    except Exception:
        print("ℹ️ Prophet unavailable or failed — using linear trend fallback.")
        y_prophet = _linear_trend(df, horizon)

    # XGBoost
    y_xgb = None
    try:
        y_xgb = _call_if_available("src.models.xgb_model", "xgb_predict", df, horizon, city=city)
        y_xgb = np.asarray(y_xgb, dtype=float)
    except Exception:
        print("ℹ️ XGBoost unavailable or failed — using moving average fallback.")
        y_xgb = _moving_average(df, horizon)

    # Ensure shapes
    def _ensure_len(arr):
        arr = np.asarray(arr, dtype=float).flatten()
        if arr.size < horizon:
            if arr.size == 0:
                return np.array([np.nan]*horizon, dtype=float)
            return np.concatenate([arr, np.repeat(arr[-1], horizon - arr.size)])
        if arr.size > horizon:
            return arr[:horizon]
        return arr
    y_lstm = _ensure_len(y_lstm)
    y_prophet = _ensure_len(y_prophet)
    y_xgb = _ensure_len(y_xgb)
    return y_lstm, y_prophet, y_xgb

def ensemble_forecast(history_df, city=None, horizon=14):
    # copy
    df = history_df.copy()
    # filter
    if city is not None and "city" in df.columns:
        df = df[df["city"].astype(str).str.lower() == str(city).lower()].copy()

    if df.empty:
        # return empty structured DataFrame
        future = pd.date_range(pd.Timestamp.now().normalize()+pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame({
            "date": future,
            "yhat_lstm": [np.nan]*horizon,
            "yhat_prophet": [np.nan]*horizon,
            "yhat_xgb": [np.nan]*horizon,
            "lstm_weight": [np.nan]*horizon,
            "prophet_weight": [np.nan]*horizon,
            "xgb_weight": [np.nan]*horizon,
            "final_pred": [np.nan]*horizon
        })

    date_col = _get_date_col(df)
    if date_col is None:
        raise ValueError("No date column found in dataframe for ensemble_forecast.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    y_lstm, y_prophet, y_xgb = _safe_call_predictors(df, city, horizon)

    # compute weights by inverse of model proxy error (proxy: variance of preds)
    var_l = float(np.nanvar(y_lstm))
    var_p = float(np.nanvar(y_prophet))
    var_x = float(np.nanvar(y_xgb))
    # avoid zero-sum
    eps = 1e-6
    inv = np.array([1.0/(var_l+eps), 1.0/(var_p+eps), 1.0/(var_x+eps)], dtype=float)
    # if any model is constant (var close to 0), give it small weight floor
    floor = 0.05
    inv = np.maximum(inv, floor)
    if np.isfinite(inv).sum() == 0:
        w = np.array([1/3,1/3,1/3])
    else:
        w = inv / inv.sum()

    final = w[0]*y_lstm + w[1]*y_prophet + w[2]*y_xgb

    # future dates
    last_date = df[date_col].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    out = pd.DataFrame({
        "date": future_dates,
        "yhat_lstm": y_lstm,
        "yhat_prophet": y_prophet,
        "yhat_xgb": y_xgb,
        "lstm_weight": [float(w[0])]*horizon,
        "prophet_weight": [float(w[1])]*horizon,
        "xgb_weight": [float(w[2])]*horizon,
        "final_pred": final
    })
    return out

def run_full_ensemble_forecast(history_df, city=None, horizon=14):
    # backwards compatible wrapper
    return ensemble_forecast(history_df, city=city, horizon=horizon)
