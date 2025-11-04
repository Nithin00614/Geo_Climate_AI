import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_temperature_model(df):
    """
    Train a Random Forest model to predict temperature.
    Saves the trained model as a .pkl file in the models folder.
    """
    if df is None or df.empty:
        print("⚠️ No data available for training.")
        return None, None, None

    X = df[["humidity", "pressure", "wind_speed"]]
    y = df["temperature"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=8
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "temperature_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("💾 Random Forest model saved successfully!")
    print(f"📊 Mean Absolute Error: {mae:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    return model, mae, r2
