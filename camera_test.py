import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    ret, frame = cap.read()
    #cv.imshow('Iframe', frame)

    #flipped_frame = cv.flip(frame, 1)
    #cv.imshow('flipframe', flipped_frame)

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

        # Draw the bounding box and center on the closest face
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Print the center coordinates in the terminal
        print(f"({center_x}, {center_y})")

    cv.imshow('FaceDetect', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows