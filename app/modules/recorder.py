import csv
import os
import threading
import time
import numpy as np
from pathlib import Path
from .features import identify_keyframes, FEATURE_SIZE
import app.config as cfg

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "gestures.csv"
HEADER = ["class", "sequence_id"] + [f"f_{i}" for i in range(FEATURE_SIZE)]

def ensure_csv(path: Path = None):
    p = path or DATA_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)

class RecordingSession:
    def __init__(self, gesture_name: str, duration: int = 60, data_path: Path = None):
        self.gesture_name = gesture_name
        self.duration = duration
        self._data_path = data_path or DATA_PATH
        self.landmarks = []
        self.status = "idle"   # idle -> recording -> processing -> done -> error
        self.progress = 0.0    # 0-100
        self.message = ""
        self.frames_saved = 0
        self._start_time = None
        self._lock = threading.Lock()

    def add_landmark(self, lm: np.ndarray):
        if self.status == "recording":
            with self._lock:
                self.landmarks.append(lm)
            elapsed = time.time() - self._start_time
            self.progress = min(100.0, (elapsed / self.duration) * 100)
            if elapsed >= self.duration:
                self.stop()

    def start(self):
        self.status = "recording"
        self._start_time = time.time()
        self.progress = 0.0

    def stop(self):
        if self.status != "recording":
            return
        self.status = "processing"
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        try:
            ensure_csv(self._data_path)
            with self._lock:
                all_lm = list(self.landmarks)

            if len(all_lm) < 3:
                self.status = "error"
                self.message = "Not enough frames captured."
                return

            arr = np.array(all_lm)
            keyframes = identify_keyframes(arr)

            import pandas as pd
            if self._data_path.exists() and self._data_path.stat().st_size > 0:
                df = pd.read_csv(self._data_path)
                next_id = int(df['sequence_id'].max()) + 1 if len(df) > 0 else 1
            else:
                next_id = 1

            frame_window = cfg.FRAME_WINDOW
            rows_written = 0
            with open(self._data_path, 'a', newline='') as f:
                writer = csv.writer(f)
                seq_id = next_id
                for kf in keyframes:
                    start_idx = max(0, kf - frame_window)
                    end_idx = min(len(arr), kf + frame_window + 1)
                    for idx in range(start_idx, end_idx):
                        writer.writerow([self.gesture_name, seq_id] + list(arr[idx]))
                        rows_written += 1
                    seq_id += 1

            self.frames_saved = rows_written
            self.status = "done"
            self.progress = 100.0
            self.message = f"Saved {rows_written} rows for '{self.gesture_name}'"
        except Exception as e:
            self.status = "error"
            self.message = str(e)

# Global session
current_session: RecordingSession = None
