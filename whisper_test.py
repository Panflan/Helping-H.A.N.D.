import whisper
import sounddevice as sd
import numpy as np
import wave
import time
import keyboard
import os
import sys

Sample_rate = 16000 #16kHz

model = whisper.load_model("base")

def record_audio(filename="temp.wav"):
    print("Now listening... Press 's' to stop and translate. \n")
    q = []
    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.append(indata.copy())
    
    with sd.InputStream(samplerate=Sample_rate, channels=1, dtype=np.float32, callback=callback):
        while not keyboard.is_pressed('s'):
            time.sleep(0.1)
    print("\nRecording stopped...")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(Sample_rate)
        wf.writeframes((np.concatenate(q) * 32767).astype(np.int16).tobytes())

    return filename    

"""
I intend to have handsigns indicating stop and record speech
for now I need to be able to translate it, either might be the code or smtg
"q" to quit the program (0)
"r" to record voice (1)
"s" to stop recording voice (2)
"""

def transcribe_audio():
    filename = record_audio()

    if not os.path.exists(filename):
        print("Error: Audio file is not found.")
        return
    
    print("Transcribing...(beep boop)")
    result = model.transcribe(filename)
    transcription = result["text"] 

    print("Text: ",transcription)
    return transcription
    