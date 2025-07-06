import cv2
import mediapipe as mp
import numpy as np
import os

# Define gesture labels here
labels = ["Hello", "Thank You", "Sorry", "Yes", "I Love You"]
X = []
y = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"\n✋ Show gesture: '{label}' (Hold for 100 frames)")
    input("➡️  Press Enter when ready...")

    count = 0
    while count < 100:
        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract features
                landmarks = hand_landmarks.landmark
                features = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
                X.append(features)
                y.append(label)
                count += 1
                cv2.putText(img, f"{label}: {count}/100", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Collecting Data", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save the dataset
os.makedirs("data", exist_ok=True)
np.save("data/X_data.npy", np.array(X))
np.save("data/y_labels.npy", np.array(y))

print("✅ Gesture data collection complete and saved.")
