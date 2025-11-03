import pandas as pd
import matplotlib.pyplot as plt
import os

def get_data_path(filename="weather_data_clean.csv"):
    """Get absolute path for any file in the /data directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", filename)

def plot_temperature_trends(df):
    """
    Plot temperature trends for each city from the cleaned dataset.
    """
    if df is None or df.empty:
        print("⚠️ Not enough data to plot yet. Try fetching more samples.")
        return

    if "date_time" not in df.columns or "city" not in df.columns:
        print("⚠️ Missing required columns for plotting.")
        return

    # Convert date_time to datetime if not already
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # Create a folder to save plots
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot temperature trend for each city
    cities = df["city"].unique()
    for city in cities:
        city_df = df[df["city"] == city].sort_values("date_time")
        plt.figure(figsize=(8, 4))
        plt.plot(city_df["date_time"], city_df["temperature"], marker="o", linestyle="-")
        plt.title(f"🌡️ Temperature Trend for {city}")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, f"{city}_trend.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"✅ Temperature trend plots saved in: {plots_dir}")

if __name__ == "__main__":
    data_path = get_data_path()
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        plot_temperature_trends(df)
    else:
        print("❌ No cleaned data file found! Run the fetch & preprocess steps first.")
