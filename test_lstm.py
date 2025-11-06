from src.lstm_model import train_lstm_model, predict_next_7_days
import pandas as pd

df = pd.read_csv("data/weather_data.csv")
model, scaler = train_lstm_model(df, "Delhi")
predict_next_7_days(df, "Delhi", scaler)
