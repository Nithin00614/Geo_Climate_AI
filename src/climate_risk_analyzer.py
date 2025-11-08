import pandas as pd

def compute_climate_risk(df):
    df["risk"] = "Low"
    df.loc[df["predicted_temperature"] > 35, "risk"] = "High"
    df.loc[(df["predicted_temperature"] > 30) & (df["predicted_temperature"] <= 35), "risk"] = "Moderate"
    return df
