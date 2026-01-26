import pandas as pd
import os
from prophet import Prophet
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

DATA_PATH = "data/processed/"
MODEL_PATH = "src/models/trained/"
os.makedirs(MODEL_PATH, exist_ok=True)

def train(city_file):
    city = city_file.replace(".csv", "")
    df = pd.read_csv(os.path.join(DATA_PATH, city_file))
    
    df["date"] = pd.to_datetime(df["date"])
    df["ds"] = df["date"]
    df["y"] = df["temp"]

    # Prophet Model
    prophet_model = Prophet()
    prophet_model.fit(df[["ds", "y"]])
    joblib.dump(prophet_model, os.path.join(MODEL_PATH, f"{city}_prophet.pkl"))

    # XGBoost Model
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["temp"].values
    xgb = XGBRegressor(n_estimators=150)
    xgb.fit(X, y)
    joblib.dump(xgb, os.path.join(MODEL_PATH, f"{city}_xgb.pkl"))

    # LSTM Model
    seq = 30
    values = y.reshape(-1, 1)
    sequences = []
    labels = []
    
    for i in range(len(values) - seq):
        sequences.append(values[i:i+seq])
        labels.append(values[i+seq])
    
    X_lstm = np.array(sequences)
    y_lstm = np.array(labels)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0)
    model.save(os.path.join(MODEL_PATH, f"{city}_lstm.h5"))

    print(f"✔ Training completed: {city}")


if __name__ == "__main__":
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            train(file)
