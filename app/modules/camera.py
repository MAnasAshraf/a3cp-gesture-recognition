class FrameProcessor:
    """Lightweight state container for landmark routing.

    MediaPipe now runs client-side in the browser (Tasks JS).
    This class only holds callback/mode state used by recording and inference.
    The name 'processor' is kept for backward compatibility with main.py references.
    """

    def __init__(self):
        self.landmark_callback = None
        self.mode = "preview"          # "preview" | "recording" | "inference"
        self.prediction_overlay = None

    def start(self):
        pass

    def stop(self):
        pass


processor = FrameProcessor()
