import os
import numpy as np
import joblib
from extract_features import extract_mfcc
from sklearn.ensemble import RandomForestClassifier

X = []
y = []

# REAL = 0
for file in os.listdir("dataset/real"):
    path = f"dataset/real/{file}"
    feat = extract_mfcc(path)
    X.append(feat)
    y.append(0)

# FAKE = 1
for file in os.listdir("dataset/fake"):
    path = f"dataset/fake/{file}"
    feat = extract_mfcc(path)
    X.append(feat)
    y.append(1)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "tts_detector.pkl")
print("✅ Model trained & saved as tts_detector.pkl")
