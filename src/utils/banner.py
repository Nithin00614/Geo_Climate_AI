import streamlit as st
from datetime import datetime


def top_banner():
    """Render top banner with title and project info."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
            padding: 16px 24px;
            border-bottom: 2px solid #64748b;
            margin: -70px -16px 20px -16px;
            border-radius: 0;
        ">
            <h2 style="color: #f1f5f9; margin: 0 0 4px 0;">
                🌍 GeoClimate AI
            </h2>
            <p style="color: #cbd5e1; margin: 0; font-size: 13px;">
                AI-Powered Weather Forecasting & Climate Risk Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
