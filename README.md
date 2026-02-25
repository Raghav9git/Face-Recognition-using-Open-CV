# Face Recognition System using Python and OpenCV

## Overview

This project implements a complete real-time Face Recognition System using Python and OpenCV.

The system performs:

1. Face dataset collection using webcam  
2. Model training using LBPH algorithm  
3. Real-time face detection  
4. Real-time face recognition with name labeling  
5. Unknown face detection  
6. Confidence score display in terminal  

The system uses:
- Haar Cascade Classifier for face detection
- LBPH (Local Binary Pattern Histogram) for face recognition

This project demonstrates a classical computer vision-based face recognition pipeline without deep learning.

---

# Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer

Install required dependencies:

```bash
pip install opencv-python numpy
```

---

# COMPLETE STEP-BY-STEP PROCESS

This project is divided into 4 main stages:

1. Dataset Collection
2. Model Training
3. Real-Time Face Detection
4. Real-Time Face Recognition

---

# STEP 1: DATASET COLLECTION

## Objective
Capture face images from webcam and store them for training.

## Process

1. Start webcam using OpenCV.
2. Convert frame to grayscale.
3. Detect faces using Haar Cascade.
4. Crop detected face region.
5. Save cropped face images into dataset folder.
6. Capture 100–200 images per person.
7. Press 'q' to stop capturing.

## Flow

Webcam → Capture Frame → Convert to Gray → Detect Face → Crop Face → Save Image → Repeat

## Output

dataset/
    Person_Name_1/
    Person_Name_2/

Each folder contains face images of one person.

---

# STEP 2: MODEL TRAINING (LBPH)

## Objective
Train the face recognition model using collected dataset.

## Process

1. Read all images from dataset folder.
2. Convert images to grayscale.
3. Assign numeric labels to each person.
4. Store images in features list.
5. Store labels in labels list.
6. Convert lists into NumPy arrays.
7. Train LBPHFaceRecognizer.
8. Save trained model as face_trained.yml.
9. Save features.npy and labels.npy.

## Flow

Read Dataset → Extract Faces → Assign Labels → Convert to NumPy → Train LBPH → Save Model

## Files Generated

- face_trained.yml
- features.npy
- labels.npy

---

# STEP 3: REAL-TIME FACE DETECTION

## Objective
Detect face from live webcam feed.

## Process

1. Start webcam.
2. Convert frame to grayscale.
3. Apply Haar Cascade classifier.
4. Get face coordinates (x, y, w, h).
5. Draw rectangle around face.

## Flow

Webcam → Frame → Gray → Haar Cascade → (x, y, w, h) → Draw Rectangle

## Output

Green bounding box around detected face.

---

# STEP 4: REAL-TIME FACE RECOGNITION

## Objective
Recognize trained faces and differentiate unknown faces.

## Process

1. Detect face using Haar Cascade.
2. Crop face region.
3. Pass cropped face to trained LBPH model.
4. Model predicts:
   - Label ID
   - Confidence score
5. Compare confidence with threshold.
6. If confidence < threshold:
       Display person name.
   Else:
       Display "Unknown".
7. Draw green bounding box.
8. Display name above rectangle.
9. Print confidence score in terminal.
10. Press 'q' to exit.

## Recognition Logic

Detected Face → Predict (label, confidence)

If confidence < threshold:
    Recognized
Else:
    Unknown

---

# CONFIDENCE SCORE EXPLANATION

Lower confidence value = Better match  
Higher confidence value = Poor match  

Example:

Confidence = 30 → Strong match  
Confidence = 85 → Weak match  

Threshold Example:

If confidence < 70:
    Recognized
Else:
    Unknown

Confidence score is printed in terminal for monitoring accuracy.

---

# COMPLETE PROJECT EXECUTION ORDER

1. Run data collection script
2. Capture images for each person
3. Run training script
4. Model gets trained and saved
5. Run Test_Model.py
6. Webcam opens
7. Face detected
8. Model predicts label
9. Green box drawn
10. Name displayed (if trained)
11. Unknown displayed (if not trained)
12. Confidence printed in terminal
13. Press 'q' to exit

---

# HOW TO RUN

Ensure the following files are in same directory:

- face_trained.yml
- haar_face.xml
- features.npy
- labels.npy
- Collect_Data.py
- Train_Model.py
- Test_Model.py

Run:

```bash
python Test_Model.py
```

Press 'q' to exit.

---

# OUTPUT

- Live webcam feed
- Green bounding box around face
- Recognized name displayed
- "Unknown" label for untrained faces
- Confidence score printed in terminal

---

# LEARNING OUTCOMES

- Understanding Haar Cascade detection
- Understanding LBPH feature extraction
- Real-time video processing
- Classical ML-based face recognition
- Confidence threshold tuning
- Unknown face classification

---

# FUTURE IMPROVEMENTS

- Add GUI interface
- Improve lighting normalization
- Add database integration
- Replace LBPH with deep learning model (CNN / FaceNet)
- Deploy as desktop application

---

This project is built to understand classical real-time computer vision-based face recognition using OpenCV and Python.
