import numpy as np
import cv2 as cv
from whisper_test import transcribe_audio
import threading

'''GoogleGenai code'''

from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv(dotenv_path="API.env") 

API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def process_speech():
    transcription = transcribe_audio()
    if not transcription or not transcription.strip():
        print("No trancription avaiable.")
        return
    
    print("Generating response from API...")
    response = model.generate_content(transcription)
    print("API Response:", response.text)

'''OpenCV code'''

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

    if closest_face:
        x, y, w, h = closest_face
        center_x = x + w // 2
        center_y = y + h // 2

        overlay = frame.copy()
        opacity = 0
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (192, 192, 192), 2)
        cv.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        
        cv.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
    cv.imshow('FaceDetect', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        threading.Thread(target=process_speech, daemon=True).start()  # Run recording in background

cap.release()
cv.destroyAllWindows()



"""
ASL detection, finger signs and face recognition
link to whisper_test.py, know when to start/stop recording voice
Add timer(extra)
"""