import joblib
import numpy as np
from audio_utils.chunk_audio import split_audio
from extract_features import extract_mfcc

model = joblib.load("tts_detector.pkl")

file = input("Enter long audio path: ")

chunks, sr = split_audio(file)

fake_scores = []
for i, chunk in enumerate(chunks):
    feat = extract_mfcc(file)
    proba = model.predict_proba([feat])[0][1]
    fake_scores.append(proba)

final_fake = np.mean(fake_scores) * 100

print(f"AI Generated (Fake): {final_fake:.2f}%")
print(f"Real Human Voice: {100-final_fake:.2f}%")
