import librosa
import numpy as np

def extract_mfcc(input_data, sr=16000, from_array=False):
    if from_array:
        y = input_data
    else:
        y, sr = librosa.load(input_data, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)
