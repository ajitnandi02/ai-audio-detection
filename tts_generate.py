from gtts import gTTS

texts = [
    "Hello, this is an AI generated voice",
    "This speech is created using text to speech"
]

for i, text in enumerate(texts, 1):
    tts = gTTS(text=text, lang='en')
    tts.save(f"tts{i}.mp3")
