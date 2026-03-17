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
| `app/modules/features.py` | Landmark extraction → flat numpy array |
| `app/modules/recorder.py` | Captures landmark frames during recording |
| `app/modules/trainer.py` | LSTM training on movement gesture CSV |
| `app/modules/recognizer.py` | Sliding window LSTM inference (movement) |
| `app/modules/audio_recorder.py` | Receives browser PCM chunks, extracts MFCCs |
| `app/modules/audio_trainer.py` | LSTM training on audio gesture CSV |
| `app/modules/audio_recognizer.py` | Rolling buffer LSTM inference (audio) |
| `app/static/index.html` | Entire frontend — single-page app |

## Data
- `data/gestures.csv` — movement gesture training data (MediaPipe landmarks)
- `data/audio_gestures.csv` — audio gesture training data (MFCC features, 43-dim)
- `data/models/movement_model.h5` + `label_encoder.pkl`
- `data/models/audio_model.h5` + `audio_label_encoder.pkl`

## Known constraints
- Each gesture class needs **at least 2 recordings** before training (stratified split)
- Audio recording uses browser mic via WebSocket — macOS blocks direct mic access
  from Python processes, so the browser captures and streams audio to the backend
- MediaPipe is not thread-safe — a single ThreadPoolExecutor(max_workers=1) is
  used for all frame processing
- The movement model uses keyframe detection (velocity/acceleration thresholds)
  rather than raw frames — see `features.py` → `identify_keyframes()`

## Frontend notes
- Single HTML file (`index.html`) — no build step, no npm, no framework
- Camera ping-pong: browser sends JPEG at ≤15fps, server returns JSON landmarks
- Audio: `AudioContext({ sampleRate: 22050 })` → `ScriptProcessorNode(4096)` → WebSocket
- Skeleton drawn with Canvas2D on a transparent overlay (video always live underneath)
- Fusion prediction: `/api/fusion/prediction` combines movement + audio with
  confidence-weighted logic (agreement bonus, discount when models disagree)
