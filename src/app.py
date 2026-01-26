# ================================================
# GeoClimate AI - Main App Launcher (No Import Errors)
# ================================================

import os
import sys
import streamlit as st

# ------------------------------------------------
# FIX PYTHON PATH (WORKS FROM ANY LOCATION)
# ------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now src folder becomes importable:
# src/models
# src/utils
# src/pages  (Streamlit autoloads)
# ------------------------------------------------

# Set page config
st.set_page_config(
    page_title="GeoClimate AI",
    page_icon="🌍",
    layout="wide",
)

# Import common UI elements
try:
    from src.utils.sidebar_style import apply_sidebar_style
    from src.utils.banner import top_banner
except Exception as e:
    st.error(f"⚠️ UI import error: {e}")

# Apply sidebar + banner
try:
    apply_sidebar_style()
    top_banner()
except:
    pass

# Home page content
st.title("🌍 GeoClimate AI — Smart Climate Intelligence Platform")

st.markdown("""
### 👋 Welcome to GeoClimate AI  
A unified AI-powered system for:
- **14-Day Ensemble Weather Forecasts**
- **IoT Sensor Monitoring & Alerting**
- **Climate Risk Analytics**
- **Model Explainability (XAI)**
- **Automated Retraining**
- **Daily Summary Emails**
- **Database Viewer**
""")

