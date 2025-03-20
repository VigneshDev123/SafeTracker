import serial
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Serial communication setup
arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)


def sendMail():
    message = MIMEMultipart("alternative")
    message["Subject"] = 'Notification regarding e-challan fine'
    message["From"] = 'Your_mail_id'
    message["To"] = 'Sender_mail_id'
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    body = f'You have been detected using a high beam in a well-lit area, which is a violation of traffic rules. A challan of ₹1000 has been issued against your vehicle.  Please make the payment at the official traffic department portal to avoid further penalties.'

    message.attach(MIMEText(body, "plain"))
    server.login('Your_mail_id', 'your_password')
    server.sendmail('Your_mail_id', 'Sender_mail_id', message.as_string())
    server.quit()
def detect_high_beam():
    while True:
        if arduino.in_waiting > 0:
            light_intensity = int(arduino.readline().decode('utf-8').strip())
            print(f"Light Intensity: {light_intensity}")

            if light_intensity < 795:  # High beam detected
                print("High Beam Violation Detected! Sending Email...")
                sendMail()
                break  # End process after sending mail

detect_high_beam()
