# Face Recognition System using OpenCV (Classical CV)

A real-time face recognition system built with classical computer vision —
Haar Cascade for face **detection** and LBPH (Local Binary Pattern Histograms)
for face **recognition**. No deep learning; this project demonstrates the
classical CV pipeline end-to-end and its real-world trade-offs.

## What it does
- Collects face images from a webcam and labels them
- Trains an LBPH recognizer on the collected faces
- Runs live recognition, labeling known faces and flagging unknown ones
- Prints a confidence (distance) score per frame for monitoring

## Detection vs. Recognition
- **Detection** (Haar Cascade): finds *where* a face is in the frame.
- **Recognition** (LBPH): identifies *whose* face it is.

## Tech stack
- Python 3
- opencv-contrib-python  (the `contrib` build is required for `cv2.face`)
- NumPy

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

> Note: install **opencv-contrib-python**, not opencv-python — the LBPH
> recognizer (`cv2.face.LBPHFaceRecognizer_create`) lives in the contrib module.

## How to run (in order)
1. `python 01_collect_faces.py` — capture ~100 face crops into `datasets/`
2. `python 02_train_model.py` — train LBPH, produces `models/face_trained.yml`
3. `python 03_recognize_live.py` — live recognition with name labels

Press `q` to quit any webcam window.

## Understanding the confidence score
LBPH confidence is a **distance, not a percentage** — **lower means a better match.**
Typical values for a correctly recognized face are ~20–40; the script labels a
face as "Unknown" when the distance exceeds the threshold.

## Known limitations (and why)
- **Lighting sensitivity:** LBPH compares raw pixel-intensity texture, so a
  lighting change between training and testing raises the distance and degrades
  accuracy.
- **Scale sensitivity:** trained on faces at one rough size, so recognition is
  best at a similar distance from the camera.
- **Per-frame flicker:** each frame is classified independently with no
  temporal smoothing, so borderline matches flip between the name and "Unknown."

## Possible improvements
- Smooth predictions across frames (majority vote over N frames)
- Replace Haar with a DNN face detector (more robust to angle/lighting)
- Replace LBPH with face embeddings (FaceNet / dlib) for far better accuracy
- Multi-person support with a label→name mapping file and attendance logging

## A second experiment: `celebrity_demo/`
A parallel version that recognizes a small set of public figures from static
image folders, using the same Haar + LBPH approach. Included to contrast
recognition on curated webcam data vs. uncontrolled internet images — the
latter produces much weaker matches, illustrating LBPH's data-quality dependence.
