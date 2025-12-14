import sounddevice as sd
import soundfile as sf

duration = 5      # seconds
samplerate = 16000

print("🎙️ Speak now...")
audio = sd.rec(int(duration * samplerate),
               samplerate=samplerate,
               channels=1)
sd.wait()

sf.write("data/sample.wav", audio, samplerate)
print("✅ sample.wav created successfully")
