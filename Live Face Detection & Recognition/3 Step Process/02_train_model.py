"""import cv2
import numpy as np
from PIL import Image
import os   

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "datasets"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []    
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L') 
        faceNP = np.array(faceImage, 'uint8')
        Id = int(os.path.split(imagePaths)[-1].split(".")[1])
        faces.append(faceNP)
        ids.append(Id)
        cv2.imshow("Training",faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write("face_trained.yml")
cv2.destroyAllWindows()
print("Training completed...........")    

"""

# TrainingModel.py

"""import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "datasets"

def getImageID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # convert to grayscale
        faceNP = np.array(faceImage, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])  # convert to int!
        faces.append(faceNP)
        ids.append(id)
        print(f"[INFO] Loaded image for ID {id}: {imagePath}")
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)
    return ids, faces

ids, faces = getImageID(path)
recognizer.train(faces, np.array(ids))
recognizer.save("face_trained.yml")
cv2.destroyAllWindows()
print("Training completed.")
"""



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
