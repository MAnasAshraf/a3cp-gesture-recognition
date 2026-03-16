import cv2
import mediapipe as mp
import numpy as np
from .features import extract_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


class FrameProcessor:
    """
    Stateless-ish MediaPipe processor.
    The browser captures webcam frames and sends JPEG bytes via WebSocket.
    We process them here and return annotated JPEG bytes.
    Camera permission is handled entirely by the browser — no OS-level
    camera access is needed by this backend process.
    """

    def __init__(self):
        self._holistic = None
        self.landmark_callback = None   # set when recording/inference is active
        self.mode = "preview"           # "preview" | "recording" | "inference"
        self.prediction_overlay = None  # text drawn on inference frames

    def start(self):
        if self._holistic is None:
            self._holistic = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def stop(self):
        if self._holistic:
            self._holistic.close()
            self._holistic = None

    def process(self, jpeg_bytes: bytes) -> bytes:
        """Decode → MediaPipe → draw landmarks → re-encode. Returns JPEG bytes."""
        if self._holistic is None:
            return jpeg_bytes

        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jpeg_bytes

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        rgb.flags.writeable = True

        # Draw face mesh
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )
        # Draw left hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )
        # Draw pose skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

        # Inference overlay text
        if self.mode == "inference" and self.prediction_overlay:
            cv2.putText(
                image, self.prediction_overlay, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (99, 102, 241), 3, cv2.LINE_AA,
            )

        # Feed landmarks to recorder / recognizer
        lm = extract_landmarks(results)
        if lm is not None and self.landmark_callback:
            self.landmark_callback(lm)

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()


processor = FrameProcessor()
