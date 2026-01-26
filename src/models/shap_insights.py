import numpy as np
import pandas as pd

# fake example feature importance for demonstration
def get_xgb_feature_importance():
    return pd.DataFrame({
        "feature": ["temperature", "humidity", "pressure", "wind_speed"],
        "importance": [0.42, 0.33, 0.15, 0.10]
    })

def get_model_contributions(row):
    return {
        "LSTM": row["yhat_lstm"],
        "Prophet": row["yhat_prophet"],
        "XGBoost": row["yhat_xgb"],
    }
