# A3CP – Multimodal Assistive Communication

## What this project is
A browser-based assistive communication tool built for The Open University.
Users record custom gestures (movement) and sounds (audio), train personalised
LSTM neural networks, and then use live recognition to communicate in real-time.
The system fuses movement and audio predictions using confidence-weighted logic.

## How to run
```bash
cd /Users/ma3289/Downloads/A3CP/a3cp
/Users/ma3289/Downloads/A3CP/gesture_env/bin/python run.py
```
Then open http://localhost:8000 in the browser.

## Python environment
- Location: `/Users/ma3289/Downloads/A3CP/gesture_env/`
- Python 3.11 (required — TF mutex crash on 3.9)
- Key versions: tensorflow-macos==2.13.0, mediapipe==0.10.9, numpy==1.24.3
- These versions are tightly coupled — do not upgrade without checking compatibility

## Architecture
```
Browser (getUserMedia) → JPEG → WebSocket /ws/camera → MediaPipe (server)
                                                      → JSON landmarks → Browser draws skeleton client-side

Browser (AudioContext 22050Hz) → Float32 PCM → WebSocket /ws/audio
                                             → recording session OR audio recognizer
```

### Why landmarks are drawn client-side
The original approach sent annotated JPEGs back from the server — this caused
significant display lag (full JPEG encode/decode round-trip per frame). Now the
server returns JSON landmark positions and the browser draws the skeleton on a
transparent canvas overlay, keeping the live video smooth at all times.

### Why AudioContext is set to 22050 Hz
The audio models are trained at SAMPLE_RATE=22050. Setting the browser
AudioContext to the same rate avoids any resampling.

### Why Python 3.11 and TF 2.13
- Python 3.9 + TF 2.20 causes a mutex crash on macOS (libc++abi termination)
- mediapipe 0.10.9 requires protobuf <4; TF 2.14+ requires protobuf >=4
- TF 2.13 + mediapipe 0.10.9 + numpy 1.24.3 is the only tested compatible set

## Module overview
| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, all HTTP + WebSocket endpoints |
| `app/modules/camera.py` | MediaPipe Holistic, returns JSON landmarks |
| `app/modules/features.py` | Landmark extraction → centroid-normalized 1657-dim vector + `compute_deltas()` |
| `app/modules/recorder.py` | Captures continuous landmark sequences (no keyframe shredding) |
| `app/modules/trainer.py` | LSTM training on movement gesture CSV |
| `app/modules/recognizer.py` | Sliding window LSTM inference (movement) |
| `app/modules/face_trainer.py` | LSTM training on face landmarks only |
| `app/modules/face_recognizer.py` | Sliding window LSTM inference (face) |
| `app/modules/audio_recorder.py` | Receives browser PCM chunks, extracts MFCCs |
| `app/modules/audio_trainer.py` | LSTM training on audio gesture CSV |
| `app/modules/audio_recognizer.py` | Rolling buffer LSTM inference (audio) |
| `app/static/index.html` | Entire frontend — single-page app |

## Feature vector layout (1657-dim)
```
Pose:        cols   0–98    (33 landmarks × 3)
Left hand:   cols  99–161   (21 landmarks × 3)
Left angles: cols 162–175   (14 joint angles)
Right hand:  cols 176–238   (21 landmarks × 3)
Right angles:cols 239–252   (14 joint angles)
Face:        cols 253–1656  (468 landmarks × 3)
```
Constants exported from `features.py`:
- `FEATURE_SIZE = 1657` (base vector stored in CSV)
- `DELTA_FEATURE_SIZE = 253` (body-column deltas appended at training/inference)
- `FULL_FEATURE_SIZE = 1910` (what the LSTM actually sees: 1657 + 253)
- `FACE_COL_START = 253`, `FACE_COL_END = 1657`
- `LEFT_ANGLE_COLS = np.arange(162, 176)`
- `RIGHT_ANGLE_COLS = np.arange(239, 253)`

### Centroid normalization
All x,y coordinates are normalized by subtracting the mid-shoulder position
(average of pose landmarks 11 and 12). This makes the model invariant to where
the user stands in the camera frame. If pose is lost, the last known mid-shoulder
is cached and reused to prevent coordinate jumps.

### Delta features (velocity)
`compute_deltas(sequence)` appends frame-to-frame differences for body columns
(0–252) to each frame. The LSTM sees both position and velocity, making it easier
to distinguish dynamic gestures (clap, wave) from static poses (idle).

### Recording: continuous sequences
Each recording is saved as one continuous sequence (single `seq_id`), truncated
to `MAX_SEQUENCE_LENGTH = 60` frames (≈4s at 15fps). No keyframe shredding —
the LSTM sees the full temporal flow of the gesture.

**BREAKING CHANGE (2026-03-18):** Right hand added; FEATURE_SIZE changed 1580 → 1657.
Any `movement_model.h5` or `face_model.h5` trained before this date must be deleted
and retrained from scratch with new recordings.

**BREAKING CHANGE (2026-03-22):** Audio features expanded from 43-dim (mean only) to
121-dim (mean + max + std for MFCC/delta/delta2 + 4 spectral). Captures transient
sounds (e.g. claps) that were previously washed out by mean-only averaging.
Any `audio_model.h5`, `audio_scaler.pkl`, `audio_selector.pkl`, and
`audio_gestures.csv` data must be deleted and re-recorded/retrained.

**BREAKING CHANGE (2026-03-22, v2):** Full architectural reset — centroid normalization,
delta features (velocity columns), continuous sequences (no keyframe shredding),
fixed `MAX_SEQUENCE_LENGTH = 60`. LSTM input is now `(batch, 60, 1910)`.
All `gestures.csv`, `movement_model.h5`, `face_model.h5` must be deleted and
re-recorded/retrained from scratch.

## Data
Per-user paths (feature/multi-user branch):
- `data/users/{username}/gestures.csv` — movement gesture data
- `data/users/{username}/audio_gestures.csv` — audio gesture data (MFCC, 43-dim)
- `data/users/{username}/models/movement_model.h5` + `label_encoder.pkl`
- `data/users/{username}/models/face_model.h5` + `face_label_encoder.pkl`
- `data/users/{username}/models/audio_model.h5` + `audio_label_encoder.pkl`
- `data/users/{username}/models/audio_scaler.pkl` + `audio_selector.pkl`

Legacy fallback paths (if no username supplied):
- `data/gestures.csv`, `data/audio_gestures.csv`, `data/models/`

## Fusion approach
Confidence-based heuristic (no meta-learner):
- `THETA_AUD = 0.90` — audio overrides everything if confidence ≥ threshold
- `THETA_VIS = 0.85` — movement wins at this confidence
- `THETA_MIN = 0.20` — minimum confidence for any output (below = "no gesture")
- `W_AGREE = 1.10`, `W_DISAGREE = 0.90` — agreement bonus / disagreement penalty
- Face is **modifier only** — never a primary candidate. Can boost winner ×1.05 if it agrees
  (suppressed when hands are active via `HAND_VEL_THRESH = 0.02`)
- Endpoint: `GET /api/fusion/prediction` — response includes `hand_active` boolean

## Audio pipeline
- 121-dim features (MFCC mean/max/std + delta mean/max/std + delta2 mean/max/std + 4 spectral)
- → StandardScaler → SelectKBest(k=14) → LSTM(64) → Dense(32,relu) → Dense(n,softmax)
- Scaler and selector saved alongside model and loaded during inference
- Max/std statistics added to capture transient sounds (claps, taps) that mean-only averaging missed

## Known constraints
- Each gesture class needs **at least 2 recordings** before training (stratified split)
- Audio recording uses browser mic via WebSocket — macOS blocks direct mic access
  from Python processes, so the browser captures and streams audio to the backend
- MediaPipe is not thread-safe — a single ThreadPoolExecutor(max_workers=1) is
  used for all frame processing
- Movement recordings are saved as continuous sequences (max 60 frames ≈ 4s)
- Delta features (velocity) are computed on-the-fly during training and inference

## Frontend notes
- Single HTML file (`index.html`) — no build step, no npm, no framework
- Camera ping-pong: browser sends JPEG at ≤15fps, server returns JSON landmarks
- Audio: `AudioContext({ sampleRate: 22050 })` → `ScriptProcessorNode(4096)` → WebSocket
- Skeleton drawn with Canvas2D on a transparent overlay (video always live underneath)
- Fusion prediction: `/api/fusion/prediction` combines movement + audio + face with
  confidence-weighted heuristic (agreement bonus, discount when models disagree)
