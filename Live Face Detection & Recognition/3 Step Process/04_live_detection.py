import cv2 as cv

cap = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')   # load once, before the loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)                        # normalize contrast

    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    print(f'Number of faces found = {len(faces_rect)}')

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    cv.imshow('Detected Face', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
