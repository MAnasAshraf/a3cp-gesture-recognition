"""
app.config

Runtime configuration and constants for A3CP.

Edit this file to change default values.
Thresholds and colours can also be changed live from the Settings page in the UI
(POST /api/config/apply) — no server restart needed for those parameters.
Audio and camera parameters require a server restart to take effect.

Constraints:
- constants only
- no runtime logic
- no filesystem IO
"""

# ── Fusion heuristic  (matches researcher notebook cell 42) ──────────────────

THETA_AUD  = 0.90   # audio overrides all streams at or above this confidence
THETA_VIS  = 0.85   # movement wins as primary at or above this confidence
W_AGREE    = 1.10   # confidence multiplier when two streams agree
W_DISAGREE = 0.90   # confidence multiplier when streams disagree

# ── Recognition ───────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.50   # minimum confidence to report any prediction
PREDICTION_INTERVAL  = 0.3    # seconds between LSTM inference calls
WINDOW_SIZE          = 11     # sliding window length (frames fed to movement LSTM)

# ── Audio pipeline ────────────────────────────────────────────────────────────

SAMPLE_RATE     = 22050  # Hz — must match browser AudioContext sampleRate (restart required)
N_MFCC          = 13     # MFCC coefficients extracted per window
WINDOW_DURATION = 0.5    # seconds per analysis window
HOP_DURATION    = 0.1    # seconds between consecutive windows
KBEST_K         = 14     # SelectKBest feature count for audio model (matches notebook cell 30)

# ── Camera and MediaPipe  (restart required) ──────────────────────────────────

FPS_CAP                        = 15     # max frames sent to server per second
JPEG_QUALITY                   = 0.75   # browser JPEG encode quality (0.0–1.0)
MEDIAPIPE_MODEL_COMPLEXITY     = 1      # 0 = lite, 1 = full, 2 = heavy
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE  = 0.5

# ── Training defaults ─────────────────────────────────────────────────────────

EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
TEST_SIZE     = 0.2    # fraction of data reserved for validation

# ── Keyframe detection  (movement recording) ──────────────────────────────────

VELOCITY_THRESHOLD     = 0.1
ACCELERATION_THRESHOLD = 0.1
FRAME_WINDOW           = 5    # frames around each keyframe to include

# ── Landmark skeleton colours  (CSS rgba strings) ─────────────────────────────
# Can be changed live from the Settings page.

COLOR_LEFT_HAND  = "rgba(160,40,180,0.9)"
COLOR_RIGHT_HAND = "rgba(245,100,30,0.9)"
COLOR_POSE       = "rgba(245,117,66,0.85)"
COLOR_FACE_OVAL  = "rgba(80,200,80,0.6)"
