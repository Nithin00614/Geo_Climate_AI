import os
import joblib
import numpy as np

def predict_temperature(humidity=None, pressure=None, wind_speed=None):
    """
    Predict temperature using the saved model.
    Accepts named arguments: humidity, pressure, wind_speed
    """

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "data", "models", "temperature_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at {model_path}")

        # Load the trained model
        model = joblib.load(model_path)

        # Prepare features
        if None in [humidity, pressure, wind_speed]:
            raise ValueError("Please provide all inputs: humidity, pressure, and wind_speed")

        features = np.array([[humidity, pressure, wind_speed]])

        # Make prediction
        prediction = model.predict(features)[0]

        print(f"🤖 Predicted Temperature: {prediction:.2f} °C")
        return prediction

    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        return None
