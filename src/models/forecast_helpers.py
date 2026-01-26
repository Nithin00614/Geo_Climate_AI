"""
forecast_helpers.py
Safe wrappers for calling model-specific forecast functions (LSTM, Prophet, XGBoost).
This file intentionally tries multiple import names and provides fallbacks so pages don't crash.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import importlib
import traceback

# ---------- Utilities ----------
def _try_import(module_path: str, fn_name: str) -> Optional[Callable]:
    """Try to import fn_name from module_path; return callable or None."""
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn
    except Exception:
        return None
    return None

def _log(msg: str):
    # simple print so Streamlit logs show it
    print(msg)

# ---------- LSTM wrapper ----------
def run_lstm_forecast(history_df: pd.DataFrame, horizon: int = 14, **kwargs) -> pd.Series:
    """
    history_df: DataFrame with a date column and a numeric target column (temperature)
    horizon: days to predict
    Returns: pandas Series indexed by future dates of length horizon
    """
    # try several common function names
    candidates = [
        ("src.models.lstm_model", "lstm_predict_next"),
        ("src.models.lstm_model", "predict_next_lstm"),
        ("src.models.lstm_model", "predict_lstm"),
        ("models.lstm_model", "lstm_predict_next"),
        ("lstm_model", "lstm_predict_next"),
    ]
    for module, fn in candidates:
        fn_obj = _try_import(module, fn)
        if fn_obj:
            try:
                _log(f"✅ Calling LSTM function: {module}.{fn}")
                out = fn_obj(history_df.copy(), horizon=horizon, **kwargs)
                # Expect out as Series or array-like
                s = pd.Series(out).reset_index(drop=True)
                return _ensure_indexed_series(s, history_df, horizon)
            except Exception as e:
                _log(f"⚠️ LSTM call failed ({module}.{fn}): {e}\n{traceback.format_exc()}")

    _log("⚠️ LSTM model not available — using persistence fallback.")
    return _persistence_forecast(history_df, horizon)

# ---------- Prophet wrapper ----------
def run_prophet_forecast(history_df: pd.DataFrame, horizon: int = 14, **kwargs) -> pd.Series:
    candidates = [
        ("src.models.prophet_model", "prophet_forecast"),
        ("src.models.prophet_model", "run_prophet_forecast"),
        ("models.prophet_model", "prophet_forecast"),
        ("prophet_model", "prophet_forecast"),
    ]
    for module, fn in candidates:
        fn_obj = _try_import(module, fn)
        if fn_obj:
            try:
                _log(f"✅ Calling Prophet function: {module}.{fn}")
                out = fn_obj(history_df.copy(), horizon=horizon, **kwargs)
                s = pd.Series(out).reset_index(drop=True)
                return _ensure_indexed_series(s, history_df, horizon)
            except Exception as e:
                _log(f"⚠️ Prophet call failed ({module}.{fn}): {e}\n{traceback.format_exc()}")

    _log("⚠️ Prophet model not available — using simple trend fallback.")
    return _linear_trend_forecast(history_df, horizon)

# ---------- XGBoost wrapper ----------
def run_xgb_forecast(history_df: pd.DataFrame, horizon: int = 14, **kwargs) -> pd.Series:
    candidates = [
        ("src.models.xgb_model", "xgb_forecast"),
        ("src.models.xgb_model", "predict_xgb"),
        ("models.xgb_model", "xgb_forecast"),
        ("xgb_model", "xgb_forecast"),
    ]
    for module, fn in candidates:
        fn_obj = _try_import(module, fn)
        if fn_obj:
            try:
                _log(f"✅ Calling XGBoost function: {module}.{fn}")
                out = fn_obj(history_df.copy(), horizon=horizon, **kwargs)
                s = pd.Series(out).reset_index(drop=True)
                return _ensure_indexed_series(s, history_df, horizon)
            except Exception as e:
                _log(f"⚠️ XGBoost call failed ({module}.{fn}): {e}\n{traceback.format_exc()}")

    _log("⚠️ XGBoost model not available — using persistence fallback.")
    return _persistence_forecast(history_df, horizon)

# ---------- Fallback helper forecasts ----------
def _persistence_forecast(history_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return last observed value repeated for horizon days"""
    if history_df.empty:
        return pd.Series([np.nan] * horizon)
    # try common temp column names
    value = _get_last_value(history_df)
    return pd.Series([value] * horizon)

def _linear_trend_forecast(history_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Estimate a linear trend from last 7 points and extend it."""
    if history_df.shape[0] < 2:
        return _persistence_forecast(history_df, horizon)
    # get numeric series
    y = _get_numeric_series(history_df)
    x = np.arange(len(y))
    # simple linear regression
    a, b = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + horizon)
    preds = a * future_x + b
    return pd.Series(preds)

def _ensure_indexed_series(s: pd.Series, history_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Convert plain series s to indexed series with future dates matching history_df frequency."""
    # determine last date
    date_col = _find_date_col(history_df)
    if date_col is None:
        idx = pd.RangeIndex(start=0, stop=horizon)
        s.index = idx
        return s
    last_date = pd.to_datetime(history_df[date_col].iloc[-1])
    freq = _infer_freq(history_df, date_col) or 'D'
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'),
                                 periods=horizon, freq='D')
    s.index = future_dates
    s.index.name = 'date'
    return s

def _get_last_value(df: pd.DataFrame) -> float:
    # try common columns
    for col in ['y', 'temperature', 'temp', 'value']:
        if col in df.columns:
            val = df[col].dropna()
            if not val.empty:
                return float(val.iloc[-1])
    # fallback: last numeric column
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        col = numeric.columns[-1]
        val = numeric[col].dropna()
        if not val.empty:
            return float(val.iloc[-1])
    return float(np.nan)

def _get_numeric_series(df: pd.DataFrame) -> np.ndarray:
    for col in ['y', 'temperature', 'temp', 'value']:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors='coerce').dropna().values
            if arr.size:
                return arr
    numeric = df.select_dtypes(include=[np.number]).dropna()
    if not numeric.empty:
        return numeric.iloc[:, -1].values
    return np.array([np.nan])

def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if 'date' in c.lower() or c.lower() in ('ds', 'timestamp', 'time'):
            return c
    return None

def _infer_freq(df: pd.DataFrame, date_col: str) -> Optional[str]:
    try:
        idx = pd.to_datetime(df[date_col])
        return pd.infer_freq(idx)
    except Exception:
        return None
