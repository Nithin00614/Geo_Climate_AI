# src/pages/model_manager_dashboard.py
import streamlit as st
import os
import pandas as pd
import joblib
import time
from datetime import datetime
from lstm_model import train_lstm_model
from weather_fetcher import fetch_weather_data

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

st.set_page_config(layout="wide")

st.markdown("""
# 🧪 Model Manager  
Train, retrain, delete, or inspect models for all cities.
""")
st.title("🧩 GeoClimate AI — Model Lifecycle Manager")

# --- Helper functions ---
def get_model_metadata():
    records = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".csv"):
            city = os.path.splitext(f)[0]
            data_path = os.path.join(DATA_DIR, f)
            model_path = os.path.join(MODELS_DIR, f"lstm_{city}.keras")
            scaler_path = os.path.join(MODELS_DIR, f"scaler_{city}.pkl")

            data_size = os.path.getsize(data_path) / 1024 if os.path.exists(data_path) else 0
            model_status = "✅" if os.path.exists(model_path) else "❌"
            last_modified = None
            if os.path.exists(model_path):
                last_modified = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")

            records.append({
                "City": city.capitalize(),
                "Model Status": model_status,
                "Data Size (KB)": round(data_size, 2),
                "Last Trained": last_modified or "—",
                "Model Path": model_path if os.path.exists(model_path) else "—",
                "Scaler Path": scaler_path if os.path.exists(scaler_path) else "—",
            })
    return pd.DataFrame(records)

def retrain_model(city):
    """Retrains model for one city."""
    df_path = os.path.join(DATA_DIR, f"{city.lower()}.csv")
    if not os.path.exists(df_path):
        st.warning(f"No data found for {city}, fetching new data...")
        df = fetch_weather_data(city)
        if df is None or df.empty:
            st.error(f"❌ Could not fetch new data for {city}.")
            return
        df.to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)

    with st.spinner(f"Training model for {city}..."):
        try:
            model, scaler = train_lstm_model(df, city)
            if model:
                st.success(f"✅ Model retrained successfully for {city}.")
            else:
                st.error(f"⚠️ Model training failed for {city}.")
        except Exception as e:
            st.error(f"❌ Training error for {city}: {e}")

def delete_model(city):
    """Deletes model & scaler for one city."""
    model_path = os.path.join(MODELS_DIR, f"lstm_{city.lower()}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{city.lower()}.pkl")

    deleted = []
    for path in [model_path, scaler_path]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
    if deleted:
        st.warning(f"🗑️ Deleted model files for {city}.")
    else:
        st.info(f"No model files found for {city}.")

# --- Main dashboard ---
if st.button("🔄 Refresh Model Status"):
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


df = get_model_metadata()
if df.empty:
    st.warning("⚠️ No city data available.")
    st.stop()

st.success(f"✅ {len(df)} cities loaded.")
st.dataframe(df, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    selected_city = st.selectbox("🏙️ Choose a city to manage", sorted(df["City"].unique()))

with col2:
    if st.button("🧠 Retrain Model"):
        retrain_model(selected_city)
        time.sleep(2)
        st.experimental_rerun()

with col3:
    if st.button("🗑️ Delete Model"):
        delete_model(selected_city)
        time.sleep(2)
        st.experimental_rerun()

st.markdown("---")
st.download_button(
    "📥 Export Model Metadata",
    df.to_csv(index=False).encode("utf-8"),
    "model_metadata.csv",
    "text/csv",
)
