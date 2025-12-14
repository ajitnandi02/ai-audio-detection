import os
import numpy as np
import joblib
from sklearn.svm import SVC
from extract_features import extract_mfcc

DATASET = "deepfake_dataset"
MODEL_PATH = "tts_detector.pkl"

X, y = [], []

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATASET, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            features = extract_mfcc(os.path.join(folder_path, file))
            X.append(features)
            y.append(label)

model = SVC(kernel="rbf", probability=True)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print("✅ TTS Detection model trained & saved")
