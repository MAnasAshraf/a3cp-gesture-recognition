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

CONFIDENCE_THRESHOLD = 0.50   # minimum confidence to report a prediction
ANGLE_COLS = slice(162, 176)  # columns normalized by /180 during training

class Recognizer:
    def __init__(self, window_size=11):
        self._default_window_size = window_size
        self.model = None
        self.le = None
        self.window = deque(maxlen=window_size)
        self.prediction = {"class": "—", "confidence": 0.0}
        self.running = False
        self._lock = threading.Lock()
        self._last_predict_time = 0.0

    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Train a model first.")
        self.model = load_model(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)
        
        # Dynamically set window size based on trained model's expected shape
        try:
            expected_length = self.model.input_shape[1]
            if expected_length is not None:
                with self._lock:
                    self.window = deque(maxlen=expected_length)
        except Exception:
            pass

    def is_loaded(self):
        return self.model is not None

    def add_frame(self, lm: np.ndarray):
        snapshot = None
        with self._lock:
            self.window.append(lm)
            if self.window.maxlen and len(self.window) >= self.window.maxlen and (time.time() - self._last_predict_time) >= 0.3:
                self._last_predict_time = time.time()
                snapshot = list(self.window)  # copy before releasing lock
        if snapshot is not None:
            self._predict(snapshot)

    def _predict(self, snapshot):
        # Runs outside the lock so get_prediction() is never blocked by TF inference
        seq = np.array(snapshot, dtype='float32')
        # Apply the same angle normalization used during training (trainer.py line 66)
        seq[:, ANGLE_COLS] /= 180.0
        X = pad_sequences([seq], padding='post', dtype='float32', value=-1.0)
        probs = self.model.predict(X, verbose=0)[0]
        idx = np.argmax(probs)
        confidence = float(probs[idx])
        result = (
            {"class": self.le.classes_[idx], "confidence": confidence}
            if confidence >= CONFIDENCE_THRESHOLD
            else {"class": "—", "confidence": confidence}
        )
        with self._lock:
            self.prediction = result

    def get_prediction(self):
        with self._lock:
            return dict(self.prediction)

recognizer = Recognizer()
