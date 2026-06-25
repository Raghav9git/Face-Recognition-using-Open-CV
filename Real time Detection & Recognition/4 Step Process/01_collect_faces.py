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
