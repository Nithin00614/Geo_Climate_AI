import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def train_lstm_model(df, city_name):
    """
    Train an LSTM model for a given city's temperature forecast.
    Saves both model and scaler for future predictions.
    """
    city_df = df[df['city'].str.lower() == city_name.lower()]

    if len(city_df) < 10:
        print(f"⚠️ Not enough data for {city_name} (need at least 10 samples).")
        return None, None

    city_df = city_df.sort_values("date_time")

    # Prepare features
    data = city_df[['temperature', 'humidity', 'pressure', 'wind_speed']].values

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    seq_len = 3
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i:i+seq_len])
        y.append(scaled_data[i+seq_len][0])
    X, y = np.array(X), np.array(y)

    # Define model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    model.fit(X, y, epochs=80, batch_size=8, verbose=1, callbacks=[early_stop])

    # ✅ Save model first
    model_path = os.path.join("models", "lstm_temperature_model.keras")
    model.save(model_path)
    print(f"✅ Model saved at: {model_path}")

    # ✅ Force save scaler using absolute path
    try:
        scaler_path = os.path.abspath(os.path.join("models", "scaler.pkl"))
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved successfully at: {scaler_path}")
    except Exception as e:
        print("❌ Error saving scaler:", e)
        print("➡️ Retrying with open() method...")
        with open(scaler_path, "wb") as f:
            joblib.dump(scaler, f)
        print("✅ Scaler saved via fallback method.")

    return model, scaler


def predict_next_7_days(df, city_name, scaler):
    """
    Predict next 7 days' temperature for a given city using trained LSTM model.
    """
    model_path = os.path.join("models", "lstm_temperature_model.keras")
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Please train it first.")
        return None

    model = load_model(model_path)

    city_df = df[df['city'].str.lower() == city_name.lower()].sort_values("date_time")

    if len(city_df) < 3:
        print(f"⚠️ Not enough data for {city_name} to make predictions.")
        return None

    # Prepare last sequence
    last_seq = city_df[['temperature', 'humidity', 'pressure', 'wind_speed']].values[-3:]
    last_seq_scaled = scaler.transform(last_seq)
    input_seq = np.expand_dims(last_seq_scaled, axis=0)

    predictions = []
    for _ in range(7):
        pred_scaled = model.predict(input_seq)
        next_input = np.append(input_seq[:, 1:, :], [[
            [pred_scaled[0][0], last_seq_scaled[-1][1], last_seq_scaled[-1][2], last_seq_scaled[-1][3]]
        ]], axis=1)
        input_seq = next_input
        predictions.append(pred_scaled[0][0])

    dummy = np.zeros((len(predictions), 4))
    dummy[:, 0] = predictions
    inv_predictions = scaler.inverse_transform(dummy)[:, 0]

    print(f"\n🌤️ 7-Day Temperature Forecast for {city_name.capitalize()}:")
    for i, temp in enumerate(inv_predictions, 1):
        print(f"Day {i}: {temp:.2f} °C")

    return inv_predictions


if __name__ == "__main__":
    print("\n🌦️ LSTM Temperature Forecast Trainer")
    city_name = input("Enter city name to train model for: ")

    data_path = os.path.join("data", "weather_data.csv")

    if not os.path.exists(data_path):
        print("❌ Data file not found! Make sure weather_data.csv exists in the /data folder.")
    else:
        df = pd.read_csv(data_path)
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

        print(f"\n📊 Training LSTM model for {city_name}...")
        model, scaler = train_lstm_model(df, city_name)

        if model is not None and scaler is not None:
            print(f"\n✅ Model and scaler saved successfully for {city_name}!")
        else:
            print("⚠️ Training failed. Try using a city with more data points.")
