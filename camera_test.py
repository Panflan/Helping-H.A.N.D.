import numpy as np
import cv2 as cv
from whisper_test import transcribe_audio
import threading
import numpy as np
import mediapipe as mp
import pickle
import time

# GoogleGenai

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
    
    print("Generating response from API...\n")
    response = model.generate_content(transcription)
    print("API Response:", response.text)


# OpenCV

cap = cv.VideoCapture(2)
'''
 numbers represent which camera or device, 0 is for front 
'''
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'Q', 1: 'R', 2: 'S'}

last_predicted = None
last_printed_time = 0
delay_seconds = 3 

def character_delay(character):
    global last_printed_time
    print(character)
    last_printed_time = time.time()
    time.sleep(delay_seconds)

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

        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (192, 192, 192), 2)        
        cv.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
    '''
    Credits
    
    Method adapted from @computervisioneng via https://github.com/computervisioneng/sign-language-detector-python/
    '''

    H, W, _ = frame.shape
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_, y_  = [], []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
                )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
    
        current_time = time.time()
        if predicted_character != last_predicted and (current_time - last_printed_time) > delay_seconds:
            threading.Thread(target=character_delay, args=(predicted_character),daemon=True).start() 
            last_predicted = predicted_character # the line that updates the last predicted character

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv.putText(frame, predicted_character, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)

    cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF
    if last_predicted == 'Q':
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