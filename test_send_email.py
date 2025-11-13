import smtplib
from email.mime.text import MIMEText

sender = "nithin00614@gmail.com"
password = "mdvq uala iroj lise"  # your app password
receiver = "nithingowda00614@gmail.com"

msg = MIMEText("✅ GeoClimate AI test email — Gmail App Password setup successful.")
msg["Subject"] = "GeoClimate AI Alert Test"
msg["From"] = sender
msg["To"] = receiver

try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Email failed:", e)
