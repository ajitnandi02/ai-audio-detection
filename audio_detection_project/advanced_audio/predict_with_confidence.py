import joblib
from extract_features import extract_mfcc

model = joblib.load("tts_detector.pkl")

file = input("Enter audio file: ")

feat = extract_mfcc(file)
proba = model.predict_proba([feat])[0]

fake = proba[1] * 100
real = proba[0] * 100

if fake > 70:
    confidence = "HIGH CONFIDENCE FAKE"
elif fake > 55:
    confidence = "LIKELY FAKE"
else:
    confidence = "LIKELY REAL"

print(f"Fake: {fake:.2f}% | Real: {real:.2f}%")
print(f"Verdict: {confidence}")
