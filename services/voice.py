# services/voice.py
import speech_recognition as sr

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        print("Processing audio...")
        return recognizer.recognize_google(audio)
