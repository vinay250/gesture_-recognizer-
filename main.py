import cv2
import mediapipe as mp
import numpy as np
from gesture_recognizer import GestureRecognizer
from utils import extract_hand_features, speak_text

recognizer = GestureRecognizer()
recognizer.load_model()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_label = ""

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_hand_features(hand_landmarks.landmark)

            try:
                label = recognizer.predict(features)
                cv2.putText(img, f'Gesture: {label}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                if label != prev_label:
                    speak_text(label)
                    prev_label = label
            except:
                cv2.putText(img, "Unrecognized Gesture", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
