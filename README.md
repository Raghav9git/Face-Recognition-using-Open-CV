# Face Recognition System using Python and OpenCV

## Overview

This project implements a real-time Face Recognition System using Python and OpenCV. The system captures facial data using a webcam, trains a recognizer model, and performs live face recognition with identity labeling.

The implementation uses Haar Cascade for face detection and the LBPH (Local Binary Pattern Histogram) algorithm for face recognition.

---

## Features

- Real-time face detection using Haar Cascade classifier
- Face dataset collection from webcam
- Model training using LBPHFaceRecognizer
- Real-time face recognition with confidence score
- Unknown face detection handling
- Live bounding box and label display

---

## Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer

---

## Working Principle

1. Face Detection  
   The Haar Cascade classifier detects faces in real-time video frames.

2. Data Collection  
   Face images are captured from the webcam and stored for training purposes.

3. Training Phase  
   The LBPH algorithm extracts facial features and trains a recognition model.  
   The trained model is saved as `face_trained.yml`.

4. Recognition Phase  
   During live execution:
   - Face is detected
   - Model predicts label ID
   - Confidence score is calculated
   - If confidence is within threshold, identity name is displayed
   - Otherwise, the face is marked as "Unknown"

---

## How to Run

1. Install required dependencies:

pip install opencv-python numpy

2. Ensure the following files are in the same directory:
   - face_trained.yml
   - haar_face.xml
   - features.npy
   - labels.npy

3. Run the recognition system:

python Test_Model.py

Press 'q' to exit the application.

---

## Output

- Live webcam feed
- Bounding box around detected faces
- Recognized name label
- Confidence score displayed in terminal
- Unknown face detection support

---

## Sample Results

The repository includes demo screenshots and video showing:
- Face data collection
- Model training
- Real-time recognition
- Unknown face detection

---

This project was developed to understand classical computer vision-based face recognition systems and real-time implementation using OpenCV.
