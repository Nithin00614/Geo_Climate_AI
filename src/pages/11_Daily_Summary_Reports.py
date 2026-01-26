import streamlit as st
import pandas as pd, os, sys
from pathlib import Path

from src.utils.sidebar_style import apply_sidebar_style
from src.utils.header import page_header
from src.utils.banner import top_banner
from src.models.forecast_engine import ensemble_forecast
from src.utils.data_loader import load_all_city_data

apply_sidebar_style()
st.set_page_config(page_title="📘 Daily Summary Reports", layout="wide")
top_banner()
page_header("Daily Summary Reports", "Generate and download daily forecast & alerts summary", "📘")

DATA_DIR = Path("data")
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

all_cities = load_all_city_data()
cities = sorted(all_cities.keys()) if all_cities else []

if not cities:
    st.error("No city data available.")
    st.stop()

city = st.selectbox("Select City for Summary", cities)

if st.button("Generate Daily Report (CSV & simple HTML)"):
    df = all_cities[city]
    
    try:
        forecast = ensemble_forecast(df, city=city, horizon=14)
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        st.stop()
    
    summary = {
        "city": city,
        "generated_at": pd.Timestamp.now().isoformat(),
        "rows": len(df),
        "forecast_start": str(forecast['date'].min()) if not forecast.empty else None,
        "forecast_end": str(forecast['date'].max()) if not forecast.empty else None,
        "mean_pred_temp": float(forecast['final_pred'].mean()) if 'final_pred' in forecast.columns and not forecast.empty else None
    }
    # CSV
    csv_path = reports_dir / f"{city.lower().replace(' ','_')}_daily_summary.csv"
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    # Simple HTML
    html = f"""
    <html><body>
    <h1>Daily Summary — {city}</h1>
    <p>Generated at: {summary['generated_at']}</p>
    <ul>
      <li>Rows: {summary['rows']}</li>
      <li>Forecast: {summary['forecast_start']} → {summary['forecast_end']}</li>
      <li>Mean predicted temp: {summary['mean_pred_temp']}</li>
    </ul>
    </body></html>
    """
    html_path = reports_dir / f"{city.lower().replace(' ','_')}_daily_summary.html"
    html_path.write_text(html)
    st.success("Report generated.")
    st.download_button("Download CSV", csv_path.read_bytes(), csv_path.name, "text/csv")
    st.download_button("Download HTML", html.encode('utf-8'), html_path.name, "text/html")
else:
    st.info("Select a city and click to generate a short daily summary report.")
