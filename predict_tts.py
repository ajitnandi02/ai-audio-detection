import joblib
from extract_features import extract_mfcc

model = joblib.load("tts_detector.pkl")

file = "dataset/fake/tts1.wav"
feat = extract_mfcc(file)

pred = model.predict([feat])[0]

if pred == 1:
    print("🤖 AI GENERATED (TTS / Fake)")
else:
    print("🧑 REAL HUMAN VOICE")
