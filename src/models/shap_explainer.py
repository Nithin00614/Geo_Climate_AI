# src/models/shap_explainer.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

def _normalize_city_data(df):
    """Ensure essential columns exist and create synthetic data if too small."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename temp -> temperature if needed
    if "temp" in df.columns and "temperature" not in df.columns:
        df.rename(columns={"temp": "temperature"}, inplace=True)

    # Add missing columns
    if "temperature" not in df.columns:
        df["temperature"] = np.random.uniform(25, 30, len(df))
    if "humidity" not in df.columns:
        df["humidity"] = np.random.uniform(40, 80, len(df))
    if "rainfall" not in df.columns:
        df["rainfall"] = np.zeros(len(df))

    # Clean NaNs
    df = df.dropna(subset=["temperature", "humidity", "rainfall"]).drop_duplicates()
    df.reset_index(drop=True, inplace=True)

    # 🧠 Synthetic augmentation if data is too small
    if len(df) < 20:
        n_missing = 30 - len(df)
        if n_missing > 0:
            base_temp = df["temperature"].mean() if not df.empty else 28
            base_hum = df["humidity"].mean() if not df.empty else 60
            base_rain = df["rainfall"].mean() if not df.empty else 0
            synthetic = pd.DataFrame({
                "temperature": np.random.normal(base_temp, 1.2, n_missing),
                "humidity": np.random.normal(base_hum, 5, n_missing),
                "rainfall": np.clip(np.random.normal(base_rain, 2, n_missing), 0, None),
            })
            df = pd.concat([df, synthetic], ignore_index=True)
            print(f"🔧 Added {n_missing} synthetic rows for SHAP stability.")
    return df

def train_xgb_for_shap(df, city):
    """Train a lightweight XGBoost model for SHAP visualization."""
    df = _normalize_city_data(df)
    features = ["temperature", "humidity", "rainfall"]
    X = df[features]
    y = df["temperature"].shift(-1).fillna(method="ffill")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=80,
        learning_rate=0.1,
        random_state=42,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(X_scaled, y)
    return model, X, X_scaled

def explain_city_xgb(df, city):
    """Generate SHAP values and feature importance summary plot."""
    df = _normalize_city_data(df)
    model, X, X_scaled = train_xgb_for_shap(df, city)
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Summary plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    fig = plt.gcf()
    return fig, shap_values
