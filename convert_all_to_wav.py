import os
import subprocess

SRC = "dataset"
DST = "dataset_wav"

os.makedirs(DST + "/real", exist_ok=True)
os.makedirs(DST + "/fake", exist_ok=True)

def convert(src, dst):
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-ac", "1",
        "-ar", "16000",
        dst
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for label in ["real", "fake"]:
    for f in os.listdir(f"{SRC}/{label}"):
        src = f"{SRC}/{label}/{f}"
        dst = f"{DST}/{label}/{f.replace('.mp3', '.wav')}"
        convert(src, dst)
        print("✅", dst)
