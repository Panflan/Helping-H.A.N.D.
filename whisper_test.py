import whisper
import sounddevice as sd
import numpy as np
import wave
import time
import keyboard
import os
import sys
 
Sample_rate = 16000 #16kHz
Silence_threshold = 0.01
Silence_duration = 3

model = whisper.load_model("base")

def record_audio(filename="temp.wav", duration=10):
    print("Now listening... Press 'q' to stop and translate. \n")
    recording = sd.rec(int(Sample_rate*duration), samplerate=Sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(Sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())

    return filename    

"""
I intend to have handsigns indicating stop and record speech
for now I need to be able to translate it, either might be the code or smtg
"""

def main():
    
    filename = record_audio(duration=10)
    print("Waiting for 'q' key to transcribe... ")
    while not keyboard.is_pressed('q'):
        time.sleep(0.1)
    print("Transcribing...(beep boop)")

    if not os.path.exists(filename):
        print("Error: Audio file is not found.")
        sys.exit(1)
    result = model.transcribe(filename)
    transcription = result["text"] 

    print("Programed finished. Now exiting...")
    return transcription
    
    
if __name__ == "__main__":
    result = main()
