import whisper
import sounddevice as sd
import numpy as np
import wave
import time
 
Sample_rate = 16000 #16kHz
Silence_threshold = 0.01
Silence_duration = 3

model = whisper.load_model("base")

def record_audio(filename="temp.wav", duration=10):
    print("Listening...")
    recording = sd.rec(int(Sample_rate*duration), samplerate=Sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(Sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())

    return filename    

def detect_silence(audio_data, sample_rate, threshold=Silence_threshold, duration=Silence_duration):
    chunk_size = int(sample_rate*0.2)
    silent_chunks = 0
    
    for i in range(0,len(audio_data)):
        chunk = audio_data[i:i+chunk_size]
        volume = np.linalg.norm(chunk) / np.sqrt(len(chunk))

        if volume < threshold:
            silent_chunks +=1
        else:
            silent_chunks = 0
        if silent_chunks >= (sample_rate*duration)/chunk_size:
            return True
    return False

def main():
    while True:
        filename = record_audio(duration=10)

        with wave.open(filename,"rb") as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        
        if detect_silence(audio_data, Sample_rate):
            print("Silence detected. Transcribing...")

            # Transcribe using Whisper
            result = model.transcribe(filename)
            print("\nTranscription:\n", result["text"])

            # Pause to avoid constant looping
            time.sleep(1)

if __name__ == "__main__":
    main()