import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/climate_ai.db")
print(pd.read_sql("PRAGMA table_info(iot_data);", conn))
conn.close()
