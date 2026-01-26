import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

CONFIG_PATH = "src/config_email.json"

def load_email_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Failed to load email config:", e)
        return None

def send_email(subject, html_body):
    try:
        cfg = load_email_config()
        if not cfg:
            print("❌ Email config missing.")
            return False

        sender = cfg["sender_email"]
        password = cfg["app_password"]
        receivers = [cfg["receiver_email"]] + cfg.get("additional_recipients", [])

        msg = MIMEMultipart()
        msg["From"] = f"GeoClimate AI Alerts <{sender}>"
        msg["To"] = ", ".join(receivers)
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        server = smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"])
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()

        print("📧 Email sent successfully!")
        return True

    except Exception as e:
        print("❌ Email failed:", e)
        return False
