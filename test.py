from ultralytics import YOLO
import cv2
from easyocr import Reader
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import pandas as pd
import smtplib
import os
from twilio.rest import Client
import geocoder
import re
import matplotlib.pyplot as plt
# Load YOLO model
model = YOLO(r"C:\Users\ramak\traffic-two-wheeler-monitoring\runs\detect\train2\weights\best.pt")  

# Load image
image_path = r"C:\Users\ramak\traffic-two-wheeler-monitoring\data\images\train\IMG20220811134437.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)
# Define the target class (e.g., class ID 0 for 'person' in COCO dataset)
target_class = 3

# Iterate over detected objects
for r in results:
    for box in r.boxes:
        cls = int(box.cls)  # Get class ID
        if cls == target_class:  # Check if it matches the required class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Save or display the cropped image
            cv2.imwrite(f"cropped_{cls}.jpg", cropped_image)
            cv2.imwrite(cropped_image, image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.imshow(image)
            plt.axis("off")
            plt.show()
g = geocoder.ip('me')

CAMERA_LOCATION = g.json['address']+f'. [Lat: {g.lat}, Lng:{g.lng}]'
def sendMail(mail):
    message = MIMEMultipart("alternative")
    message["Subject"] = 'Notification regarding e-challan fine'
    message["From"] = mail
    message["To"] = mail
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    body = f'You were caught riding without helmet near {CAMERA_LOCATION}, and were fined Rupees 500. If you are caught riding again without proper gear, you will be severely penalized.'

    message.attach(MIMEText(body, "plain"))
    server.login('your_mail_id', 'your_password')
    server.sendmail('Your_mail_id', mail, message.as_string())
    server.quit()
  

database = pd.read_csv('database.csv')

img = 'cropped_3.jpg'


if __name__ == '__main__':

    warnedNums = []

    
    reader = Reader(['en'])
    number = reader.readtext(img, mag_ratio=3)
    licensePlate = ""

    for i in [0, 1]:
        for item in number[i]:
            if type(item) == str:
                licensePlate += item

    licensePlate = licensePlate.replace(' ', '')
    licensePlate = licensePlate.upper()
    licensePlate = re.sub(r'[^a-zA-Z0-9]', '', licensePlate)
    print('License number is:', licensePlate)

    if licensePlate not in warnedNums:
        for index, plate in enumerate(database['Registration']):
            if licensePlate == plate:
                database.at[index, 'Due challan'] += 500
                mail = database['Email'][index]
                num = database['Phone number'][index]
                sendMail(mail)
                    #sendSMS(num)
                print(f"{database['Name'][index]} successfully notified!")
                warnedNums.append(licensePlate)
                database.to_csv('database.csv', index=True)