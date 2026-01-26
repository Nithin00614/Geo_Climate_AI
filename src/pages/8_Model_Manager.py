# src/pages/8_🧪_Model_Manager.py
"""
Model Manager — Manage trained models per city
"""

import streamlit as st
import os
import sys
import pandas as pd
import joblib
import shutil
from datetime import datetime
from pathlib import Path

from src.models.lstm_model import load_lstm_model, train_lstm_model
from src.models.xgb_model import load_xgb_model, train_xgb_model
from src.models.forecast_engine import ensemble_forecast
from src.utils.sidebar_style import apply_sidebar_style
from src.utils.header import page_header
from src.utils.banner import top_banner

apply_sidebar_style()
top_banner()
page_header("Model Manager", "Manage & Train Weather Forecast Models", "🧪")


def badge(text, color):
    return f"<span style='background:{color};padding:4px 8px;border-radius:8px;color:white;font-size:12px'>{text}</span>"
# ---------- Configuration ----------
DATA_DIR = Path("data")
MODELS_DIR = Path("src/models")
LOGS_DIR = Path("logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model file patterns (adjust if your project uses different names)
def lstm_model_path(city: str):
    key = city.lower().replace(" ", "_")
    return MODELS_DIR / f"lstm_{key}.keras"

def lstm_scaler_path(city: str):
    key = city.lower().replace(" ", "_")
    return MODELS_DIR / f"scaler_{key}.pkl"

def xgb_model_path(city: str):
    key = city.lower().replace(" ", "_")
    return MODELS_DIR / f"xgb_{key}.joblib"

def xgb_scaler_path(city: str):
    key = city.lower().replace(" ", "_")
    return MODELS_DIR / f"xgb_scaler_{key}.joblib"

# Safe imports for training/loading
try:
    from src.models.lstm_model import train_lstm_model, load_lstm_model
except Exception as e:
    train_lstm_model = None
    load_lstm_model = None

try:
    from src.models.xgb_model import train_xgb_model, load_xgb_model
except Exception as e:
    train_xgb_model = None
    load_xgb_model = None

# Utility: discover cities from data folder
def discover_cities():
    try:
        from src.utils.data_loader import load_all_city_data
        return list(load_all_city_data().keys())
    except:
        return []
    if not path.exists():
        return None, f"Data file not found: {path}"
# Utility: load city dataframe with safe parsing
def load_city_df(city):
    try:
        from src.utils.data_loader import load_all_city_data
        all_data = load_all_city_data()
        return all_data.get(city), None
    except Exception as e:
        return None, str(e)

# Health checks summary
def dataset_health(city):
    df, err = load_city_df(city)
    if df is None:
        return {"exists": False, "error": err}
    if err:
        return {"exists": True, "error": err, "rows": len(df)}
    missing = [c for c in ["temperature", "humidity", "wind_speed"] if c not in df.columns]
    last_date = df["date"].max() if "date" in df.columns else None
    return {"exists": True, "error": None, "rows": len(df), "missing_cols": missing, "last_date": str(last_date)}

# Model status check
def model_status(city):
    status = {}
    status["lstm_exists"] = (MODELS_DIR / f"lstm_{city.lower().replace(' ', '_')}.keras").exists()
    status["lstm_scaler"] = (MODELS_DIR / f"scaler_{city.lower().replace(' ', '_')}.pkl").exists()
    status["xgb_exists"] = (MODELS_DIR / f"xgb_{city.lower().replace(' ', '_')}.json").exists()
    status["xgb_scaler"] = (MODELS_DIR / f"xgb_scaler_{city.lower().replace(' ', '_')}.joblib").exists()
    return status

# Safe wrappers for actions
def safe_train_lstm(city):
    if train_lstm_model is None:
        return False, "LSTM training function not available"
    try:
        df, err = load_city_df(city)
        if df is None:
            return False, f"Could not load data: {err}"
        res = train_lstm_model(df, city)
        return True, f"LSTM training complete for {city}"
    except Exception as e:
        return False, str(e)

def safe_train_xgb(city):
    if train_xgb_model is None:
        return False, "XGB training function not available"
    try:
        df, err = load_city_df(city)
        if df is None:
            return False, f"Could not load data: {err}"
        res = train_xgb_model(df, city)
        return True, f"XGB training complete for {city}"
    except Exception as e:
        return False, str(e)

def safe_delete_models(city):
    deleted = []
    for fname in [f"lstm_{city.lower().replace(' ', '_')}.keras",
                  f"scaler_{city.lower().replace(' ', '_')}.pkl",
                  f"xgb_{city.lower().replace(' ', '_')}.json",
                  f"xgb_scaler_{city.lower().replace(' ', '_')}.joblib"]:
        p = MODELS_DIR / fname
        try:
            if p.exists():
                p.unlink()
                deleted.append(str(p.name))
        except Exception as e:
            st.warning(f"Failed to delete {p.name}: {e}")
    return deleted

def safe_test_forecast(city):
    # run ensemble forecast for a quick check
    if ensemble_forecast is None:
        return False, "Ensemble function not available"
    try:
        df, err = load_city_df(city)
        if df is None:
            return False, f"Could not load data: {err}"
        out = ensemble_forecast(df, city=city)
        return True, out.head(3).to_dict(orient="records")
    except Exception as e:
        return False, str(e)

# UI layout: left column controls, right column details
cities = discover_cities()
if not cities:
    st.warning("No city CSVs found in the data folder. Add CSV files to the `data/` folder (data/<city>.csv).")
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.subheader("Cities")
    selected_city = st.selectbox("Choose a city", cities)
    st.markdown("---")

    st.subheader("Per-city actions")
    if st.button("Train LSTM (Selected)"):
        with st.spinner("Training LSTM..."):
            ok, msg = safe_train_lstm(selected_city)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if st.button("Train XGB (Selected)"):
        with st.spinner("Training XGB..."):
            ok, msg = safe_train_xgb(selected_city)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if st.button("Retrain Both (Selected)"):
        with st.spinner("Retraining both models..."):
            ok1, m1 = safe_train_lstm(selected_city)
            ok2, m2 = safe_train_xgb(selected_city)
            if ok1:
                st.success(m1)
            else:
                st.error(m1)
            if ok2:
                st.success(m2)
            else:
                st.error(m2)

    if st.button("Delete Models (Selected)"):
        if st.confirm_button if hasattr(st, "confirm_button") else False:
            pass
        with st.spinner("Deleting model files..."):
            ok, msg = safe_delete_models(selected_city)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if st.button("Test Forecast (Selected)"):
        with st.spinner("Running quick ensemble check..."):
            ok, result = safe_test_forecast(selected_city)
            if ok:
                st.success("Forecast test succeeded — preview below.")
                st.json(result)
            else:
                st.error(f"Forecast test failed: {result}")

    st.markdown("---")
    st.subheader("Global actions")
    if st.button("Train XGB for All Cities"):
        with st.spinner("Training XGB for all cities..."):
            trained = {}
            for c in cities:
                ok, msg = safe_train_xgb(c)
                trained[c] = {"ok": ok, "msg": msg}
            st.write(trained)

    if st.button("Train LSTM for All Cities"):
        with st.spinner("Training LSTM for all cities..."):
            trained = {}
            for c in cities:
                ok, msg = safe_train_lstm(c)
                trained[c] = {"ok": ok, "msg": msg}
            st.write(trained)

    if st.button("Delete ALL Models"):
        confirm = st.radio("Confirm delete ALL model files?", ("No", "Yes"))
        if confirm == "Yes":
            with st.spinner("Deleting all model files..."):
                deleted = []
                for c in cities:
                    ok, msg = safe_delete_models(c)
                    deleted.append({c: msg})
                st.write(deleted)
        else:
            st.info("Deletion cancelled")

with right:
    st.subheader(f"Status — {selected_city}")
    # Dataset health
    health = dataset_health(selected_city)
    st.markdown("**Dataset Health**")
    if not health.get("exists", False):
        st.error(health.get("error"))
    else:
        if health.get("error"):
            st.warning(health.get("error"))
        st.write(f"Rows: {health.get('rows')}")
        if health.get("missing_cols"):
            st.warning(f"Missing columns: {health.get('missing_cols')}")
        st.write(f"Last date: {health.get('last_date')}")

    # Model status
    st.markdown("**Model Files**")
    mstatus = model_status(selected_city)
    st.write({
        "LSTM model file": "Exists" if mstatus["lstm_exists"] else "Missing",
        "LSTM scaler": "Exists" if mstatus["lstm_scaler"] else "Missing",
        "XGB model file": "Exists" if mstatus["xgb_exists"] else "Missing",
        "XGB scaler": "Exists" if mstatus["xgb_scaler"] else "Missing",
    })

    # Show preview of dataset (first 5 rows)
    st.markdown("**Data Preview (first 5 rows)**")
    preview_df, preview_err = load_city_df(selected_city)
    if preview_df is None:
        st.error(preview_err)
    else:
        if preview_err:
            st.warning(preview_err)
        st.dataframe(preview_df.head(5))

    # If models exist, show metadata where possible
    st.markdown("**Model Metadata / Metrics (if available)**")
    meta = {}
    # try to load saved metrics (common pattern: joblib metadata or separate metrics file)
    # We'll check for a companion file models/<prefix>_meta.joblib (optional)
    meta_file_lstm = MODELS_DIR / f"lstm_{selected_city.lower().replace(' ', '_')}_meta.joblib"
    meta_file_xgb = MODELS_DIR / f"xgb_{selected_city.lower().replace(' ', '_')}_meta.joblib"
    if meta_file_lstm.exists():
        try:
            meta_l = joblib.load(meta_file_lstm)
            meta["lstm"] = meta_l
        except Exception as e:
            meta["lstm"] = f"Failed to read meta: {e}"
    if meta_file_xgb.exists():
        try:
            meta_x = joblib.load(meta_file_xgb)
            meta["xgb"] = meta_x
        except Exception as e:
            meta["xgb"] = f"Failed to read meta: {e}"

    if meta:
        st.json(meta)
    else:
        st.info("No stored model metadata found. (You can save training metrics to models/<city>_meta.joblib in training scripts.)")

    st.markdown("---")
    st.subheader("Logs")
    # show last 200 lines of a simple log per city if exists
    log_path = LOGS_DIR / f"{selected_city.lower().replace(' ', '_')}.log"
    if log_path.exists():
        try:
            txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
            st.text("\n".join(txt))
        except Exception as e:
            st.write(f"Could not read log: {e}")
    else:
        st.info("No logs found for this city. Training scripts can write logs to logs/<city>.log")

st.markdown("---")
st.caption("Model Manager — Use retraining sparingly; prefer to test changes locally first. For long-running training, run scripts from terminal.")
