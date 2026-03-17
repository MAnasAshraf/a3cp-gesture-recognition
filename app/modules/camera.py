import json
import cv2
import mediapipe as mp
import numpy as np
from .features import extract_landmarks

mp_holistic = mp.solutions.holistic

class FrameProcessor:
    """
    Stateless-ish MediaPipe processor.
    The browser captures webcam frames and sends JPEG bytes via WebSocket.
    We run MediaPipe here and return JSON landmark positions so the browser
    can draw the skeleton itself on a transparent canvas overlay.
    This keeps the live video display smooth (zero server round-trip lag)
    while the skeleton overlay updates at MediaPipe's processing speed.
    """

    def __init__(self):
        self._holistic = None
        self.landmark_callback = None
        self.mode = "preview"
        self.prediction_overlay = None

    def start(self):
        if self._holistic is None:
            self._holistic = mp_holistic.Holistic(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def stop(self):
        if self._holistic:
            self._holistic.close()
            self._holistic = None

    def process(self, jpeg_bytes: bytes) -> bytes:
        """
        Decode → MediaPipe → extract landmarks → return JSON bytes.
        JSON shape: { left_hand, right_hand, pose, face_oval, prediction }
        Each is a list of [x, y] normalized (0-1) or null if not detected.
        """
        empty = json.dumps({
            "left_hand": None, "right_hand": None,
            "pose": None, "face_oval": None,
            "prediction": None
        }).encode()

        if self._holistic is None:
            return empty

        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return empty

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)

        # Extract landmark positions as [x, y] lists
        left_hand = (
            [[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark]
            if results.left_hand_landmarks else None
        )
        right_hand = (
            [[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark]
            if results.right_hand_landmarks else None
        )
        pose = (
            [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
            if results.pose_landmarks else None
        )
        # Send face oval subset (indices 10,338,297,332,284,251,389,356,454,323,
        # 361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,
        # 234,127,162,21,54,103,67,109) — standard 36-pt face contour
        FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,
                     378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,
                     162,21,54,103,67,109]
        face_oval = None
        if results.face_landmarks:
            lms = results.face_landmarks.landmark
            face_oval = [[lms[i].x, lms[i].y] for i in FACE_OVAL]

        # Feed landmarks to recorder / recognizer callback
        lm = extract_landmarks(results)
        if lm is not None and self.landmark_callback:
            self.landmark_callback(lm)

        prediction = (
            self.prediction_overlay
            if self.mode == "inference" and self.prediction_overlay
            else None
        )

        return json.dumps({
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose": pose,
            "face_oval": face_oval,
            "prediction": prediction,
        }).encode()


processor = FrameProcessor()
