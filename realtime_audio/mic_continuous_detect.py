import sys
import os

# ===== FIX PROJECT ROOT IMPORT =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
# ==================================

import sounddevice as sd
import numpy as np
import joblib
import queue

from extract_features import extract_mfcc

# ================= CONFIG =================
MODEL_PATH = "tts_detector.pkl"
SR = 16000
DURATION = 3   # seconds per chunk
# =========================================

model = joblib.load(MODEL_PATH)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def analyze_audio(audio_chunk):
    audio = audio_chunk.astype("float32") / 32768.0
    audio = audio.flatten()

    features = extract_mfcc(audio, sr=SR, from_array=True)
    prob = model.predict_proba([features])[0]

    real = prob[0] * 100
    fake = prob[1] * 100
    verdict = "REAL" if real > fake else "FAKE"

    print(f"\n🎤 Real: {real:.2f}% | 🤖 Fake: {fake:.2f}%")
    print(f"🧠 Verdict: {verdict}")

print("\n🎙️ Continuous Mic Detection Started")
print("👉 Speak normally | Press Q + Enter to STOP\n")

buffer = np.empty((0, 1), dtype=np.int16)

with sd.InputStream(
    samplerate=SR,
    channels=1,
    dtype="int16",
    callback=audio_callback
):
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()
            buffer = np.vstack((buffer, data))

            if len(buffer) >= SR * DURATION:
                analyze_audio(buffer[: SR * DURATION])
                buffer = buffer[SR * DURATION :]

        # ✅ CORRECT INDENTATION (FIXED)
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getch().lower() == b"q":
                print("\n🛑 Stopping...")
                break
