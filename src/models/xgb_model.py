# src/models/xgb_model.py
"""
xgb_model.py
Provides:
- train_xgb_model(df, city, save_dir='models/xgb')
- xgb_predict(df, horizon=14, city=None, save_dir='models/xgb')
Tries to use xgboost if available; otherwise training/predict raise and forecast_engine falls back.
"""

import os
import joblib
import numpy as np
import pandas as pd

XGB_AVAILABLE = True
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except Exception:
    XGB_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _create_tabular_features(df, lags=7):
    # Create simple lag features from last column (temperature)
    s = None
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna().reset_index(drop=True)
            break
    if s is None or s.empty:
        return None, None
    n = len(s)
    rows = []
    for i in range(lags, n):
        row = s.iloc[i-lags:i].values.tolist()
        rows.append((row, s.iloc[i]))
    if not rows:
        return None, None
    X = np.array([r for r,_ in rows])
    y = np.array([v for _,v in rows])
    return X, y

def train_xgb_model(df, city, save_dir="models/xgb"):
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost not installed. Install xgboost to train XGB.")
    _ensure_dir(save_dir)
    X, y = _create_tabular_features(df, lags=7)
    if X is None:
        raise ValueError("Not enough data to create features for XGB.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {"objective":"reg:squarederror", "eval_metric":"rmse", "eta":0.1, "max_depth":6}
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval,"val")], early_stopping_rounds=10, verbose_eval=False)
    path = os.path.join(save_dir, f"{city}.json")
    model.save_model(path)
    return path

def load_xgb_model(city, save_dir="models/xgb"):
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost not installed.")
    path = os.path.join(save_dir, f"{city}.json")
    if not os.path.exists(path):
        raise FileNotFoundError("XGBoost model not found.")
    model = xgb.Booster()
    model.load_model(path)
    return model

def xgb_predict(df, horizon=14, city=None, save_dir="models/xgb"):
    """
    Uses saved model for per-city predictions. If missing, raise so the ensemble will fallback.
    Implementation uses a rolling predict by building lag features from last known series.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost not installed.")
    if city is None:
        raise ValueError("city required for xgb_predict to locate model.")
    model_path = os.path.join(save_dir, f"{city}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("XGB model not found.")
    model = xgb.Booster()
    model.load_model(model_path)

    # Build initial lag vector
    # find series
    s = None
    for col in ['temperature','temp','y','value']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna().reset_index(drop=True)
            break
    if s is None or s.empty:
        raise ValueError("No numeric series for XGB predict.")
    
    # Calculate residual std for stochastic variation
    if len(s) > 2:
        residual_std = float(np.std(np.diff(s.values)))
    else:
        residual_std = float(np.std(s.values)) if len(s) > 0 else 1.0
    
    lags = 7
    seq = list(s.values[-lags:])
    preds = []
    for _ in range(horizon):
        X = np.array(seq[-lags:]).reshape(1,-1)
        dX = xgb.DMatrix(X)
        p = float(model.predict(dX)[0])
        # Add stochastic noise for variation
        noise = np.random.normal(0, residual_std * 0.1)
        p = p + noise
        preds.append(p)
        seq.append(p)
    return np.array(preds, dtype=float)
