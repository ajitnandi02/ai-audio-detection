import joblib
import numpy as np
from extract_features import extract_mfcc
from tkinter import Tk, filedialog

# model load
model = joblib.load("tts_detector.pkl")

# tkinter window hide
root = Tk()
root.withdraw()

print("📂 Please select an audio file (WAV only)...")

# file upload window
file_path = filedialog.askopenfilename(
    title="Select Audio File",
    filetypes=[("WAV files", "*.wav")]
)

if not file_path:
    print("❌ No file selected")
    exit()

# feature extract
feat = extract_mfcc(file_path)

# prediction probability
probs = model.predict_proba([feat])[0]
real_percent = probs[0] * 100
fake_percent = probs[1] * 100

print("\n🎧 Selected File:", file_path)
print(f"🧑 Real Voice: {real_percent:.2f}%")
print(f"🤖 AI Generated (Fake): {fake_percent:.2f}%")

if fake_percent > real_percent:
    print("🔴 FINAL RESULT: AI GENERATED AUDIO")
else:
    print("🟢 FINAL RESULT: REAL HUMAN AUDIO")
