import numpy as np
import joblib
import threading
from pathlib import Path
from collections import deque

from .audio_recorder import (
    extract_audio_features, SAMPLE_RATE, AUDIO_FEATURE_SIZE
)

MODEL_PATH   = Path(__file__).parent.parent.parent / "data" / "models" / "audio_model.h5"
ENCODER_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "audio_label_encoder.pkl"

WINDOW_SAMPLES       = int(0.5 * SAMPLE_RATE)   # 11 025 samples
HOP_SAMPLES          = int(0.1 * SAMPLE_RATE)    # 2 205 samples
CONFIDENCE_THRESHOLD = 0.50


class AudioRecognizer:
    def __init__(self):
        self.model      = None
        self.le         = None
        self.prediction = {"class": "—", "confidence": 0.0}
        self._lock      = threading.Lock()
        self._buf       = deque(maxlen=WINDOW_SAMPLES * 3)
        self._pending   = 0
        self._active    = False
        self._seq_len   = None

    def load(self):
        from tensorflow.keras.models import load_model
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No audio model found. Train an audio model first.")
        self.model = load_model(MODEL_PATH)
        self.le    = joblib.load(ENCODER_PATH)
        try:
            self._seq_len = self.model.input_shape[1]
        except Exception:
            self._seq_len = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def start(self):
        """Activate recognizer — audio chunks will be fed via add_audio_chunk()."""
        with self._lock:
            self._buf.clear()
            self._pending = 0
            self.prediction = {"class": "—", "confidence": 0.0}
        self._active = True

    def stop(self):
        self._active = False
        with self._lock:
            self.prediction = {"class": "—", "confidence": 0.0}

    def is_active(self) -> bool:
        return self._active

    def add_audio_chunk(self, chunk: np.ndarray):
        """Called by /ws/audio WebSocket handler with each Float32 audio chunk from browser."""
        if not self._active or self.model is None:
            return
        with self._lock:
            self._buf.extend(chunk)
            self._pending += len(chunk)

        if self._pending >= HOP_SAMPLES:
            with self._lock:
                self._pending = 0
                buf = list(self._buf)
            if len(buf) >= WINDOW_SAMPLES:
                window = np.array(buf[-WINDOW_SAMPLES:], dtype=np.float32)
                threading.Thread(target=self._predict, args=(window,), daemon=True).start()

    def _predict(self, audio_window: np.ndarray):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        try:
            feats = extract_audio_features(audio_window)
            if feats is None:
                return
            X = np.array([[feats]], dtype="float32")
            if self._seq_len and self._seq_len > 1:
                X = pad_sequences(X, maxlen=self._seq_len, padding="post",
                                  dtype="float32", value=0.0)
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
        except Exception:
            pass

    def get_prediction(self) -> dict:
        with self._lock:
            return dict(self.prediction)


audio_recognizer = AudioRecognizer()
