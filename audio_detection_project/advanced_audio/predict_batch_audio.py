import os
import joblib
from extract_features import extract_mfcc

model = joblib.load("tts_detector.pkl")

folder = input("Enter folder path: ")

for file in os.listdir(folder):
    if file.endswith(".wav"):
        path = os.path.join(folder, file)
        feat = extract_mfcc(path)
        pred = model.predict([feat])[0]
        label = "FAKE" if pred == 1 else "REAL"
        print(f"{file} → {label}")
