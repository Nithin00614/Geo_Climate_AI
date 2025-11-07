import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_lstm_model(df, city):
    """
    Train an LSTM model for the selected city and save model + scaler.
    """
    city_data = df[df["city"].str.lower() == city.lower()].copy()
    if "date_time" in city_data.columns:
        city_data.rename(columns={"date_time": "date"}, inplace=True)

    city_data = city_data.sort_values("date")
    values = city_data[["temperature", "humidity", "pressure", "wind_speed"]].values

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Create sequences (lookback = 3)
    X, y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i - 3:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        print(f"⚠️ Not enough data to train for {city}")
        return None, None

    # Build model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    model_path = f"models/lstm_{city.lower()}.keras"
    scaler_path = f"models/scaler_{city.lower()}.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved scaler to {scaler_path}")

    # Evaluate
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"📊 {city} → MAE: {mae:.2f}, R²: {r2:.2f}")

    return model, scaler


def predict_next_7_days(df, city, scaler):
    """
    Predict next 7 days of temperature for the selected city using trained LSTM.
    """
    city_data = df[df["city"].str.lower() == city.lower()].copy()
    if "date_time" in city_data.columns:
        city_data.rename(columns={"date_time": "date"}, inplace=True)

    city_data = city_data.sort_values("date")
    recent_data = city_data[["temperature", "humidity", "pressure", "wind_speed"]].tail(3).values

    scaled = scaler.transform(recent_data)
    X_input = np.expand_dims(scaled, axis=0)

    model_path = f"models/lstm_{city.lower()}.keras"
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found for {city}")
        return None

    model = load_model(model_path)

    preds, forecast_dates = [], pd.date_range(start=pd.to_datetime(city_data["date"].iloc[-1]), periods=8, freq="D")[1:]
    input_seq = X_input.copy()

    for _ in range(7):
        next_pred = model.predict(input_seq)[0][0]
        preds.append(next_pred)

        next_input = np.array([[next_pred, *input_seq[0, -1, 1:]]])
        input_seq = np.append(input_seq[:, 1:, :], np.expand_dims(next_input, axis=1), axis=1)

    forecast_scaled = np.zeros((len(preds), scaled.shape[1]))
    forecast_scaled[:, 0] = preds
    forecast = scaler.inverse_transform(forecast_scaled)

    result = pd.DataFrame({
        "date": forecast_dates,
        "predicted_temperature": forecast[:, 0]
    })
    return result

    from tqdm import tqdm

def auto_train_all_cities(df):
    """
    Automatically trains and saves LSTM models for all cities in the dataset.
    """
    cities = df["city"].unique()
    print(f"🚀 Auto-training models for {len(cities)} cities...")
    for city in tqdm(cities):
        try:
            train_lstm_model(df, city)
        except Exception as e:
            print(f"⚠️ Skipped {city}: {e}")
    print("✅ All available city models trained and saved.")

