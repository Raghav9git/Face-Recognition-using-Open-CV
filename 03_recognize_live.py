"""import cv2 as cv  
import numpy as np
#from PIL import Image
import os  

cap = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('haar_face.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

name_list = ["Raghav"]
#imgBackground = cv2.imread("background.jpg")

while True:
    ret, frame = cap.read() 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 50:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
            cv.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)
            cv.rectangle(frame, (x,y-40), (x+w, y), (50,50,255), -1)
            cv.putText(frame, name_list[serial], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2) 
        else:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)
            cv.rectangle(frame, (x,y-40), (x+w, y), (50,50,255), -1)
            cv.putText(frame, "Unknown", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2) 

        frame = cv.resize(frame, (640, 480))
        #imgBackground[162:162+480, 55:55+640] = frame
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
         """



"""# TestModel.py

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

# Map ID to name
name_list = ["Raghav"]  # ID 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        print(f"[DEBUG] ID: {serial}, Confidence: {conf:.2f}")

        if conf < 50:  # You can adjust this threshold
            name = name_list[serial] if serial < len(name_list) else "Unknown"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""



#import time

import cv2
import numpy as np

# Load components
cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

# ID to Name mapping
name_list = ["Shilpa"]  # ID 0 = Shilpa

print("Face recognition begins. Press 'q' to exit...")

# Replace 'COM4' with the port your Bluetooth module is connected to
#try:
 #   bt = serial.Serial('COM4', 9600, timeout=1)
  #  time.sleep(2)  # Give it time to connect
   # print("[INFO] Bluetooth connected.")
#except Exception as e:
 #   print("[ERROR] Failed to connect to Bluetooth:", e)
  #  bt = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        serial, conf = recognizer.predict(face_roi)

        print(f"[DEBUG] ID: {serial}, Confidence: {conf:.2f}")

        if conf < 50:  
            name = name_list[serial] if serial < len(name_list) else "Unknown"
            color = (0, 255, 0)

            # Send Bluetooth signal only once when Raghav is detected
        #if name == "Raghav" and bt:
         #  bt.write(b'open\n')  # send 'open' command
          # print("[INFO] Sent 'open' command to Arduino.")
           #time.sleep(2)  # Optional delay to avoid repeated signals
         
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
