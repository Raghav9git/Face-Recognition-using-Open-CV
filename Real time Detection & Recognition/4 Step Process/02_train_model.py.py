import cv2
import os
import numpy as np

dataset_path = "datasets"
detector = cv2.CascadeClassifier('haar_face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

print("Training the model...")

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        path = os.path.join(dataset_path, file)
        gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label = int(file.split(".")[1])
        faces.append(gray_img)
        labels.append(label)

recognizer.train(faces, np.array(labels))
recognizer.save("face_trained.yml")

print("Training completed.")
