import librosa
import numpy as np

def preprocess_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = y / np.max(np.abs(y))  # normalize
    return y, target_sr
