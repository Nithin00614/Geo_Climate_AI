# File: src/models/shap_explainer.py
# Clean, final, error-free SHAP explainer for XGB

import shap
import xgboost as xgb
import pandas as pd
import numpy as np


# -----------------------------------------------------------
#   Train XGB model for SHAP explanations
# -----------------------------------------------------------
def train_xgb_for_shap(df):
    """
    Train a simple XGBoost model for SHAP feature importance.
    df must include:
        temperature, humidity, pressure, wind_speed
    """
    required = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column for SHAP training: {col}")

    # Prepare features & target
    X = df[required]
    y = df["temperature"].shift(-1).fillna(method="ffill")

    model = xgb.XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X, y)

    return model, X


# -----------------------------------------------------------
#  Produce SHAP explanation values
# -----------------------------------------------------------
def explain_xgb_model(model, X):
    """
    Return SHAP values + summary info.
    Works with XGBoost TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)

    return shap_values, shap_mean
