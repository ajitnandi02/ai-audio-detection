import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv

# Load YAMNet model
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load audio
file_path = "data/tts_fake.wav"
audio, sr = librosa.load(file_path, sr=16000, mono=True)

# Prediction
scores, _, _ = model(audio)

# Average scores over time
mean_scores = np.mean(scores, axis=0)

# Load class names
class_map = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)

with open(class_map) as f:
    reader = csv.DictReader(f)
    class_names = [row['display_name'] for row in reader]

# 🔥 MULTIPLE SOUND DETECTION
TOP_K = 3          # কয়টা sound দেখাবে
THRESHOLD = 0.2    # minimum confidence

# Sort by confidence
top_indices = np.argsort(mean_scores)[::-1]

print("\n✅ Detected Sounds:")
count = 0
for idx in top_indices:
    confidence = mean_scores[idx]
    if confidence < THRESHOLD:
        continue

    print(f"{count+1}. {class_names[idx]} ({confidence:.2f})")
    count += 1

    if count >= TOP_K:
        break

if count == 0:
    print("❌ No significant sound detected")
