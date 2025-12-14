import tensorflow_hub as hub
import tensorflow as tf
import librosa
import numpy as np

model = hub.load("https://tfhub.dev/google/yamnet/1")

def detect_sound(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    scores, _, _ = model(audio)
    mean_scores = tf.reduce_mean(scores, axis=0)
    class_map = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    )

    import csv
    with open(class_map) as f:
        names = [row["display_name"] for row in csv.DictReader(f)]

    return names[np.argmax(mean_scores)]
