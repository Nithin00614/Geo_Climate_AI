import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

def prepare_data(city):
    df = pd.read_csv("data/historical_combined.csv")
    df = df[df["city"] == city]
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Focus on temperature
    values = df["temp_max"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)

    # Prepare sequences (past 30 days → next day)
    X, y = [], []
    for i in range(30, len(scaled)):
        X.append(scaled[i-30:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm(city):
    X, y, scaler = prepare_data(city)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_{city.lower()}.keras")
    joblib.dump(scaler, f"models/scaler_{city.lower()}.pkl")
    print(f"✅ Model & scaler saved for {city}")

if __name__ == "__main__":
    for c in ["Chennai", "Bengaluru", "Delhi"]:
        train_lstm(c)
