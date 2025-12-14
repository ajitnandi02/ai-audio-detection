from fastapi import FastAPI, UploadFile, File
import shutil
import os
import joblib
import numpy as np
from extract_features import extract_mfcc

app = FastAPI(title="AI Audio Detection API")

MODEL_PATH = "tts_detector.pkl"
model = joblib.load(MODEL_PATH)

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return {"error": "Only WAV files are supported"}

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    features = extract_mfcc(temp_path).reshape(1, -1)
    prob = model.predict_proba(features)[0]

    real = round(prob[0] * 100, 2)
    fake = round(prob[1] * 100, 2)

    verdict = "REAL HUMAN AUDIO" if real > fake else "AI GENERATED AUDIO"

    os.remove(temp_path)

    return {
        "real_percentage": real,
        "fake_percentage": fake,
        "verdict": verdict
    }
