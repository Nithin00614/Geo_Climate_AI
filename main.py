import os
from src.data_loader import fetch_weather_data
from src.preprocess import preprocess_data
from src.model import train_temperature_model
from src.visualize import plot_temperature_trends, plot_actual_vs_predicted
from src.predict import predict_temperature

if __name__ == "__main__":
    print("🌦️ GeoClimate-AI: Weather Data Collector\n")

    # Step 1: Fetch live weather data
    df = fetch_weather_data()

    if df is not None:
        # Step 2: Clean and preprocess data
        processed_df = preprocess_data(df)

        if processed_df is not None:
            # Step 3: Train model
            model, mae, r2 = train_temperature_model(processed_df)

            # Step 4: Test saved model for prediction
            print("\n🌡️ Testing saved model for prediction...")
            predicted = predict_temperature(60, 1012, 3.5)
            print(f"🤖 Predicted Temperature: {predicted:.2f}°C")

            # Step 5: Generate visualizations and save plots
            print("\n📊 Generating visualizations...")

            # Ensure plots directory exists
            base_dir = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(base_dir, "data", "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plot_temperature_trends(processed_df, save_path=os.path.join(plots_dir, "temperature_trends.png"))
            plot_actual_vs_predicted(processed_df, save_path=os.path.join(plots_dir, "actual_vs_predicted.png"))

            print(f"\n✅ Plots saved in: {plots_dir}")
            print("\n🎯 All steps completed successfully!")

        else:
            print("⚠️ No processed data available.")
    else:
        print("⚠️ Data loading failed.")
