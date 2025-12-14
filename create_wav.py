import sounddevice as sd
import soundfile as sf
import os

duration = 5      # seconds
samplerate = 16000

os.makedirs("data", exist_ok=True)

print("🎙️ Speak now...")
audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype="float32"
)
sd.wait()

sf.write("data/sample.wav", audio, samplerate)
print("✅ data/sample.wav created successfully")
