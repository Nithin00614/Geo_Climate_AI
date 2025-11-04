import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

def plot_temperature_trends(df, save_path=None):
    """Plot humidity/pressure vs temperature."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df["humidity"], df["temperature"], color='b', label="Humidity vs Temp")
    plt.scatter(df["pressure"], df["temperature"], color='r', label="Pressure vs Temp")
    plt.title("Weather Data: Temperature Trends")
    plt.xlabel("Humidity / Pressure")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"💾 Saved: {save_path}")
    plt.show()


def plot_actual_vs_predicted(df, save_path=None):
    """Plot actual vs predicted temperature comparison."""
    X = df[["humidity", "pressure", "wind_speed"]]
    y = df["temperature"]

    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)

    plt.figure(figsize=(8, 5))
    plt.scatter(y, predicted, color="purple", alpha=0.6)
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title("Actual vs Predicted Temperature")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"💾 Saved: {save_path}")
    plt.show()
