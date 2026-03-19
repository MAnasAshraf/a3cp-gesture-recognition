import numpy as np
import joblib
import threading
import time
from pathlib import Path
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH   = Path(__file__).parent.parent.parent / "data" / "models" / "face_model.h5"
ENCODER_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "face_label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.50
FACE_COLS = slice(253, 1657)  # face landmark slice of the full 1657-dim landmark vector


class FaceRecognizer:
    def __init__(self, window_size=11):
        self._default_window_size = window_size
        self.model      = None
        self.le         = None
        self.window     = deque(maxlen=window_size)
        self.prediction = {"class": "—", "confidence": 0.0}
        self._probs     = None   # raw softmax array for meta-learner
        self._lock      = threading.Lock()
        self._last_predict_time = 0.0

    def load(self, model_path: Path = None, encoder_path: Path = None):
        mp = model_path or MODEL_PATH
        ep = encoder_path or ENCODER_PATH
        if not mp.exists():
            raise FileNotFoundError("No face model found. Train a face model first.")
        self.model = load_model(mp)
        self.le    = joblib.load(ep)
        try:
            expected_length = self.model.input_shape[1]
            if expected_length is not None:
                with self._lock:
                    self.window = deque(maxlen=expected_length)
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self.model is not None

    def add_frame(self, lm: np.ndarray):
        """lm is the full 1580-dim landmark vector; face cols are sliced here."""
        snapshot = None
        with self._lock:
            self.window.append(lm[FACE_COLS])
            if (self.window.maxlen and len(self.window) >= self.window.maxlen
                    and (time.time() - self._last_predict_time) >= 0.3):
                self._last_predict_time = time.time()
                snapshot = list(self.window)
        if snapshot is not None:
            threading.Thread(target=self._predict, args=(snapshot,), daemon=True).start()

    def _predict(self, snapshot):
        seq   = np.array(snapshot, dtype="float32")
        X     = pad_sequences([seq], padding="post", dtype="float32", value=-1.0)
        probs = self.model.predict(X, verbose=0)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        result = (
            {"class": self.le.classes_[idx], "confidence": conf}
            if conf >= CONFIDENCE_THRESHOLD
            else {"class": "—", "confidence": conf}
        )
        with self._lock:
            self.prediction = result
            self._probs     = probs.tolist()

    def get_prediction(self) -> dict:
        with self._lock:
            return dict(self.prediction)

    def get_probs(self) -> list:
        """Return raw softmax probabilities (list of floats) for the meta-learner."""
        with self._lock:
            return list(self._probs) if self._probs is not None else None


face_recognizer = FaceRecognizer()
