import numpy as np
import joblib
import threading
import time
from pathlib import Path
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "movement_model.h5"
ENCODER_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "label_encoder.pkl"

class Recognizer:
    def __init__(self, window_size=30):
        self.model = None
        self.le = None
        self.window = deque(maxlen=window_size)
        self.prediction = {"class": "—", "confidence": 0.0}
        self.running = False
        self._lock = threading.Lock()

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Train a model first.")
        self.model = load_model(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)

    def is_loaded(self):
        return self.model is not None

    def add_frame(self, lm: np.ndarray):
        with self._lock:
            self.window.append(lm)
            if len(self.window) >= 10:
                self._predict()

    def _predict(self):
        seq = np.array(list(self.window))
        X = pad_sequences([seq], padding='post', dtype='float32', value=-1.0)
        probs = self.model.predict(X, verbose=0)[0]
        idx = np.argmax(probs)
        self.prediction = {
            "class": self.le.classes_[idx],
            "confidence": float(probs[idx])
        }

    def get_prediction(self):
        with self._lock:
            return dict(self.prediction)

recognizer = Recognizer()
