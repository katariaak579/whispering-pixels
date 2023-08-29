import whisper
import pyaudio 
import wave 
import os
import speech_recognition as sr
from os import system
import sys
import time

# from gtts import gTTS
# from pydub.playback import play
# from pydub import AudioSegment

STOP_KEYWORD = "stop"
model = whisper.load_model("base")
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def listening():
   
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)  
    frames = []
    r = sr.Recognizer()
        
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    output_file = "recorded_audio.wav"
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    model = whisper.load_model("tiny")
    results = model.transcribe("recorded_audio.wav")
    res=results['text']
    speak(res)
    return res

def speak(text):
    if sys.platform=='darwin':
        ALLOWED_CHARS=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.,?!-_$: ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()

# si = listening()
# print(si)

# def text_to_speech(text, lang='en', output_file='output.mp3', speed=1.5):
#     tts = gTTS(text=text, lang=lang, slow=False)
#     tts.save(output_file)
#     audio = AudioSegment.from_file(output_file)
#     audio = audio.speedup(playback_speed=speed)
#     audio.export(output_file, format="mp3")
#     os.system('afplay ' + output_file)

