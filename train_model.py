from gesture_recognizer import GestureRecognizer
import numpy as np

X = np.load("data/X_data.npy")
y = np.load("data/y_labels.npy")

model = GestureRecognizer()
model.train(X, y)

print("✅ Model trained and saved.")
