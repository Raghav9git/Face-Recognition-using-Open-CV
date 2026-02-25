"""import cv2

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
id = input('Enter your id: ')
count=0


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite(f"datasets/User.{id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (60,60,255), 1)
    cv2.imshow('Face', frame)

    k=cv2.waitKey(1)
    if count>500:
        break
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()        
"""



# CaptureImages.py

"""import cv2

cap = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
id = input('Enter your ID (e.g. 0): ')
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"datasets/User.{id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 60, 255), 2)

    cv2.imshow('Capturing Faces', frame)

    k = cv2.waitKey(1)
    if k == ord('q') or count >= 100:  # Stop after 100 images or pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
print("Image capture complete.")
"""

import cv2
import os

# Initialize
face_cascade = cv2.CascadeClassifier('haar_face.xml')  # Ensure this file exists
cap = cv2.VideoCapture(0)

# Your name and ID
person_id = 0  # ID 0 = Raghav
save_dir = "datasets"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Collecting face data. Press 'q' to quit or wait for 100 samples...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(save_dir, f"user.{person_id}.{count}.jpg")
        cv2.imwrite(file_path, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

print(f"{count} face samples")
cap.release()
cv2.destroyAllWindows()
