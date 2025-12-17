import os
import sys
import wave
import json
from vosk import Model, KaldiRecognizer

def load_model(model_dir):
    model = Model(model_dir)
    return model

def transcribe_audio(model, audio_file):
    with wave.open(audio_file, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio file must be WAV format mono PCM.")
        
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.AcceptWaveform(wf.readframes(wf.getnframes()))
        res = rec.FinalResult()
        result = json.loads(res)["text"]
        return result

if __name__ == "__main__":
    model_dir = sys.argv[1]  # Replace with your model path
    audio_file = sys.argv[2]  # Replace with your audio file path

    model = load_model(model_dir)
    transcript = transcribe_audio(model, audio_file)
    print(f"Transcript: {transcript}")
