import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/climate_ai.db")

# List all tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("📋 Tables in database:")
print(tables)

# Show iot_data schema
schema = pd.read_sql("PRAGMA table_info(iot_data);", conn)
print("\n🧱 iot_data schema:")
print(schema)

# Drop old iot_data table (if needed)
try:
    conn.execute("DROP TABLE IF EXISTS iot_data;")
    print("\n✅ Dropped old iot_data table.")
except Exception as e:
    print("⚠️ Error while dropping table:", e)

conn.commit()
conn.close()
