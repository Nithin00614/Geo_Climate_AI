import os
import pandas as pd
from datetime import datetime

LOG_PATH = os.path.join("data", "alerts_log.txt")

def log_alert(city, alert_type, message):
    """
    Logs a new alert entry to data/alerts_log.txt.
    Automatically creates file if it doesn't exist.
    """
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_line = f"{timestamp} | {city} | {alert_type} | {message}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_line)

def load_alert_history(limit=200):
    """
    Reads the last `limit` alerts from the log file as a DataFrame.
    """
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=["Timestamp", "City", "Type", "Message"])

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[-limit:]  # keep only recent alerts
    records = []
    for line in lines:
        parts = line.strip().split(" | ")
        if len(parts) == 4:
            records.append({
                "Timestamp": parts[0],
                "City": parts[1],
                "Type": parts[2],
                "Message": parts[3],
            })
    return pd.DataFrame(records)
