# src/models/prophet_model.py
"""
prophet_model.py
Uses prophet if available. Exposes:
- train_prophet_model(df, city, save_dir='models/prophet')
- prophet_forecast(df, horizon=14, city=None, save_dir='models/prophet')
"""

import os
import joblib
import pandas as pd
import numpy as np

PROPHET_AVAILABLE = True
try:
    # Prophets have different names in pip; try both
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _prepare_df_for_prophet(df):
    # Prophet expects columns ds (date) and y (target)
    if 'ds' in df.columns and 'y' in df.columns:
        pdf = df[['ds','y']].copy()
        pdf['ds'] = pd.to_datetime(pdf['ds'])
        return pdf
    # Try common names
    date_col = None
    for c in df.columns:
        if 'date' in c.lower() or c.lower() in ('timestamp','time'):
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date column found for Prophet.")
    # target
    target_col = None
    for c in ['temperature','temp','y','value']:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError("No target column found for Prophet.")
    pdf = df[[date_col, target_col]].rename(columns={date_col:'ds', target_col:'y'}).copy()
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    return pdf

def train_prophet_model(df, city, save_dir="models/prophet"):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed in this environment.")
    _ensure_dir(save_dir)
    pdf = _prepare_df_for_prophet(df)
    if len(pdf) < 10:
        raise ValueError("Not enough rows for Prophet training (recommended 30+).")
    m = Prophet()
    m.fit(pdf)
    path = os.path.join(save_dir, f"{city}.pkl")
    joblib.dump(m, path)
    return path

def load_prophet_model(city, save_dir="models/prophet"):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed.")
    path = os.path.join(save_dir, f"{city}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Prophet model not found.")
    return joblib.load(path)

def prophet_forecast(df, horizon=14, city=None, save_dir="models/prophet"):
    """
    If a saved Prophet model exists for the city, use it. Otherwise raise so engine can fallback.
    Returns numpy array length=horizon.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not available.")
    if city is None:
        raise ValueError("city required to locate saved Prophet model.")
    model_path = os.path.join(save_dir, f"{city}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Prophet model not found for city.")
    model = joblib.load(model_path)
    
    # Get historical data for calculating uncertainty
    s = None
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            break
    
    # build future dataframe
    last_date = pd.to_datetime(df[[c for c in df.columns if 'date' in c.lower() or c.lower() in ('timestamp','time')][0]].iloc[-1])
    future = model.make_future_dataframe(periods=horizon, freq='D')
    fcst = model.predict(future)
    
    # take last horizon rows and add some stochastic variation
    vals = fcst['yhat'].tail(horizon).values.copy()
    
    # Add stochastic noise based on historical variation
    if s is not None and len(s) > 2:
        residual_std = float(np.std(np.diff(s.values))) * 0.15
    else:
        residual_std = 0.5
    
    for i in range(len(vals)):
        noise = np.random.normal(0, residual_std)
        vals[i] = vals[i] + noise
    
    return np.array(vals, dtype=float)
