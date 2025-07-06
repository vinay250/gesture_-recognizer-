import numpy as np
import pyttsx3

def extract_hand_features(landmarks):
    return np.array([lm.x for lm in landmarks] + [lm.y for lm in landmarks])

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

