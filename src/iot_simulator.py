import sqlite3
import random
import time
from datetime import datetime
from src.data_engine.climate_risk_full import calculate_risk_score
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib


DB = "data/climate_ai.db"
CITY = "bengaluru"

EMAIL_SENDER = "your@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your@gmail.com"


def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print("📧 Email sent!")
    except Exception as e:
        print("❌ Email error:", e)


def save_to_db(row):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS iot_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            aqi REAL,
            risk REAL,
            severity TEXT
        )
    """)
    conn.commit()

    cur.execute("""
        INSERT INTO iot_data(timestamp, city, temperature, humidity, pressure, wind_speed, aqi, risk, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["timestamp"], row["city"], row["temperature"], row["humidity"],
        row["pressure"], row["wind_speed"], row["aqi"], row["risk"], row["severity"]
    ))
    conn.commit()
    conn.close()


def simulate():
    while True:
        reading = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY,
            "temperature": random.uniform(22, 45),
            "humidity": random.uniform(10, 90),
            "pressure": random.uniform(960, 1030),
            "wind_speed": random.uniform(1, 10),
            "aqi": random.randint(20, 350)
        }

        reading["risk"], reading["severity"] = calculate_risk_score(reading)

        print("📥 New:", reading)
        save_to_db(reading)

        if reading["severity"] == "High":
            send_email(
                f"⚠ High Risk Alert — {CITY}",
                f"""
TIME: {reading['timestamp']}
CITY: {CITY}
TEMP: {reading['temperature']}
HUMIDITY: {reading['humidity']}
AQI: {reading['aqi']}
SEVERITY: {reading['severity']}
RISK SCORE: {reading['risk']}
                """
            )

        time.sleep(5)


if __name__ == "__main__":
    simulate()
