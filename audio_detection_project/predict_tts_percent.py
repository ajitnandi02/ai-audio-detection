import joblib
from extract_features import extract_mfcc

# Load trained model (UNCHANGED)
model = joblib.load("tts_detector.pkl")

# Audio path (change only this when needed)
file = "dataset/fake/tts1.wav"   # or dataset/real/real1.wav

# Feature extraction
feat = extract_mfcc(file)

# Prediction probabilities
proba = model.predict_proba([feat])[0]

real_percent = proba[0] * 100
fake_percent = proba[1] * 100

print("🔊 Audio Analysis Result:")
print(f"🧑 Real Human Voice : {real_percent:.2f}%")
print(f"🤖 AI Generated Voice : {fake_percent:.2f}%")

# Final label
if fake_percent > real_percent:
    print("✅ Final Verdict: AI GENERATED (TTS / Fake)")
else:
    print("✅ Final Verdict: REAL HUMAN VOICE")
