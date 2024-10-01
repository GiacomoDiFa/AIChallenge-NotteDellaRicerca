from gtts import gTTS
import os

def text_to_speech(text):
    res = gTTS(text=text, lang='en')
    filename = "output.mp3"
    res.save(filename)
    os.system(f"start {filename}")

if __name__ == "__main__":
    text = "Ciao, questa e' una prova del sintetizzatore vocale."
    text_to_speech(text)
