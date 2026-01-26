import streamlit as st
import pandas as pd, os, sys
from pathlib import Path
sys.path.append("src"); sys.path.append(".")

from utils.sidebar_style import apply_sidebar_style
from utils.header import page_header
from utils.banner import top_banner

apply_sidebar_style()
st.set_page_config(page_title="🗂 Database Viewer", layout="wide")
top_banner()
page_header("Database Viewer", "Inspect SQLite tables and export", "🗂")

DB = Path("data/climate_ai.db")
if not DB.exists():
    st.warning("SQLite DB not found at data/climate_ai.db. If you want DB viewer, create or move DB here.")
    st.stop()

import sqlite3
conn = sqlite3.connect(str(DB))
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
st.write("Tables:")
st.dataframe(tables)

table = st.selectbox("Select table", options=tables['name'].tolist())
if table:
    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1000", conn)
    st.dataframe(df, use_container_width=True)
    if st.button("Export full table as CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{table}.csv", "text/csv")
conn.close()
