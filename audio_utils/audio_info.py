import librosa

def get_audio_info(path):
    y, sr = librosa.load(path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "sample_rate": sr,
        "duration_sec": round(duration, 2),
        "samples": len(y)
    }
