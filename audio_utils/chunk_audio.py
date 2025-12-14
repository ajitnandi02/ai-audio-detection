import librosa

def split_audio(path, chunk_duration=3):
    y, sr = librosa.load(path, sr=16000)
    chunk_samples = chunk_duration * sr

    chunks = []
    for i in range(0, len(y), chunk_samples):
        chunks.append(y[i:i + chunk_samples])

    return chunks, sr
