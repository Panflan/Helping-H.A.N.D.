import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)) 

    closest_face = None
    max_area = 0

    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:  # Update the closest face
            max_area = area
            closest_face = (x, y, w, h)

    # If a closest face is found, track it
    if closest_face:
        x, y, w, h = closest_face
        center_x = x + w // 2
        center_y = y + h // 2

        overlay = frame.copy()
        opacity = 0
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (192, 192, 192), 2)
        cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        
        cv.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        #print(f"({center_x}, {center_y})")

    cv.imshow('FaceDetect', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows

"""
ASL detection, finger signs and face recognition
link to whisper_test.py, know when to start/stop recording voice
Add timer(extra)
"""