import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

class GestureRecognizer:
    def __init__(self):
        self.model_path = "model/gesture_knn_model.pkl"
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.labels = open("gesture_labels.txt").read().splitlines()

    def train(self, X, y):
        self.knn.fit(X, y)
        joblib.dump(self.knn, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.knn = joblib.load(self.model_path)
        else:
            print("Model not found. Train first.")

    def predict(self, landmarks):
        return self.knn.predict([landmarks])[0]
