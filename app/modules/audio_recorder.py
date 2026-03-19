import csv
import threading
import numpy as np
from pathlib import Path
from collections import deque
import app.config as cfg

SAMPLE_RATE        = cfg.SAMPLE_RATE
WINDOW_DURATION    = cfg.WINDOW_DURATION
HOP_DURATION       = cfg.HOP_DURATION
N_MFCC             = cfg.N_MFCC
AUDIO_FEATURE_SIZE = cfg.N_MFCC * 3 + 4   # mfcc + delta + delta2 + 4 spectral = 43

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "audio_gestures.csv"
HEADER    = ["class", "sequence_id"] + [f"f_{i}" for i in range(AUDIO_FEATURE_SIZE)]


def ensure_csv(path: Path = None):
    p = path or DATA_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)


def extract_audio_features(audio_window: np.ndarray, sr: int = SAMPLE_RATE):
    """Return 43-dim feature vector for one audio window, or None on failure."""
    import librosa
    if len(audio_window) < 512:
        return None
    try:
        mfcc     = librosa.feature.mfcc(y=audio_window.astype(np.float32), sr=sr, n_mfcc=N_MFCC)
        delta    = librosa.feature.delta(mfcc)
        delta2   = librosa.feature.delta(mfcc, order=2)
        centroid = librosa.feature.spectral_centroid(y=audio_window.astype(np.float32), sr=sr)
        rolloff  = librosa.feature.spectral_rolloff(y=audio_window.astype(np.float32), sr=sr)
        zcr      = librosa.feature.zero_crossing_rate(audio_window)
        rms      = librosa.feature.rms(y=audio_window.astype(np.float32))
        return np.concatenate([
            mfcc.mean(axis=1), delta.mean(axis=1), delta2.mean(axis=1),
            [float(centroid.mean()), float(rolloff.mean()),
             float(zcr.mean()),      float(rms.mean())]
        ])
    except Exception:
        return None


def process_audio_to_windows(audio: np.ndarray, sr: int = SAMPLE_RATE):
    win = int(WINDOW_DURATION * sr)
    hop = int(HOP_DURATION * sr)
    result = []
    for start in range(0, len(audio) - win + 1, hop):
        feats = extract_audio_features(audio[start:start + win], sr)
        if feats is not None:
            result.append(feats)
    return result


class AudioRecordingSession:
    """
    Receives raw Float32 audio chunks from the browser via WebSocket
    and saves MFCC features to CSV when stopped.
    """
    def __init__(self, gesture_name: str, duration: int = 10, data_path: Path = None):
        self.gesture_name = gesture_name
        self.duration     = duration
        self._data_path   = data_path or DATA_PATH
        self.status       = "idle"
        self.progress     = 0.0
        self.message      = ""
        self.frames_saved = 0
        self._chunks: list = []
        self._lock         = threading.Lock()
        self._samples_received = 0

    def start(self):
        self.status  = "recording"
        self.progress = 0.0
        self._chunks = []
        self._samples_received = 0

    def add_audio_chunk(self, chunk: np.ndarray):
        """Called by the /ws/audio WebSocket handler for each browser audio chunk."""
        if self.status != "recording":
            return
        with self._lock:
            self._chunks.append(chunk.copy())
            self._samples_received += len(chunk)
        self.progress = min(100.0, self._samples_received / (self.duration * SAMPLE_RATE) * 100)
        if self._samples_received >= self.duration * SAMPLE_RATE:
            self.stop()

    def stop(self):
        if self.status != "recording":
            return
        self.status = "processing"
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        try:
            ensure_csv(self._data_path)
            with self._lock:
                if not self._chunks:
                    raise ValueError("No audio received — check browser mic permission.")
                audio = np.concatenate(self._chunks)

            windows = process_audio_to_windows(audio)
            if not windows:
                raise ValueError("No audio windows extracted. Try a longer recording or speak louder.")

            import pandas as pd
            next_id = 1
            if self._data_path.exists() and self._data_path.stat().st_size > 50:
                df = pd.read_csv(self._data_path)
                if len(df) > 0:
                    next_id = int(df["sequence_id"].max()) + 1

            with open(self._data_path, "a", newline="") as f:
                writer = csv.writer(f)
                for feats in windows:
                    writer.writerow([self.gesture_name, next_id] + list(feats))

            self.frames_saved = len(windows)
            self.status   = "done"
            self.progress = 100.0
            self.message  = f"Saved {len(windows)} windows for '{self.gesture_name}'"
        except Exception as e:
            self.status  = "error"
            self.message = str(e)


current_session: AudioRecordingSession = None
