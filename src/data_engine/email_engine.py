import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "nithin00614@gmail.com"
APP_PASSWORD = "dpnfmmnmdacesuha"
RECEIVER_EMAIL = "nithingowda00614@gmail.com"

def send_email_alert(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("📧 Email sent successfully!")

    except Exception as e:
        print("❌ Email failed:", e)
