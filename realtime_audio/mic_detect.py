import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sounddevice as sd
from scipy.io.wavfile import write
import joblib
from extract_features import extract_mfcc
import tempfile
import os

MODEL_PATH = "tts_detector.pkl"
DURATION = 4        # seconds
SR = 16000          # sample rate

def record_audio():
    print("🎙️ Recording... Speak now")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished")
    return audio

def detect_from_mic():
    audio = record_audio()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, SR, audio)
        temp_wav = f.name

    model = joblib.load(MODEL_PATH)
    feat = extract_mfcc(temp_wav)
    prob = model.predict_proba([feat])[0]

    real = prob[0] * 100
    fake = prob[1] * 100

    print("\n📊 Live Audio Analysis")
    print(f"🎤 Real Voice : {real:.2f}%")
    print(f"🤖 AI Generated : {fake:.2f}%")

    if fake > real:
        print("🚨 FINAL RESULT: AI GENERATED (TTS / Fake)")
    else:
        print("✅ FINAL RESULT: REAL HUMAN VOICE")

    os.remove(temp_wav)

if __name__ == "__main__":
    detect_from_mic()
