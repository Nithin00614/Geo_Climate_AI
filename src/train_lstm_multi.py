import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib

# Number of past days used for prediction
TIME_STEPS = 30

def prepare_multifeature_data(city):
    df = pd.read_csv("data/historical_combined.csv")
    df = df[df["city"] == city].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Select multiple features (add/remove as needed)
    features = ["temp_max", "temp_min", "rainfall"]
    target = "temp_max"

    data = df[features]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(TIME_STEPS, len(scaled)):
        X.append(scaled[i - TIME_STEPS:i])   # all features
        y.append(scaled[i, 0])  # target is first feature: temp_max

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_multifeature_lstm(city):
    X, y, scaler = prepare_multifeature_data(city)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=25, batch_size=16, verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_{city.lower()}.keras")
    joblib.dump(scaler, f"models/scaler_{city.lower()}.pkl")

    print(f"✅ Trained and saved LSTM model for {city}")

if __name__ == "__main__":
    cities = [
        "Chennai", "Bengaluru", "Delhi", "Mumbai", "Kolkata", "Hyderabad", 
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Coimbatore", "Kochi", 
        "Patna", "Bhopal", "Visakhapatnam", "Indore", "Nagpur", 
        "Chandigarh", "Surat", "Madurai"
    ]
    for city in cities:
        train_multifeature_lstm(city)
