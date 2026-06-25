import cv2
import numpy as np


cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

name_list = ["Raghav"]  # ID 0 = Raghav

print("Face recognition begins. Press 'q' to exit...")

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
