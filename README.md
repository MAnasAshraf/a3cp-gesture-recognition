

# A3CP - Multimodal Assistive Communication

A browser-based assistive communication system that enables users to record
custom gestures (body movement, hand gestures, facial expressions) and sounds,
train personalised LSTM neural networks, and perform real-time multi-modal
recognition to communicate.

Built for The Open University.

## Features

- **Custom gesture recording** via webcam (pose, hands, face)
- **Custom audio recording** via microphone
- **Per-user model training** (movement, audio, face LSTMs)
- **Real-time inference** with confidence-weighted multi-modal fusion
- **Browser-based MediaPipe** (no server-side video processing needed)

## How to use

1. Create a user profile
2. Record gestures and/or audio samples (minimum 2 recordings per class)
3. Train your personalised models
4. Start live recognition — the system fuses movement, audio, and face predictions in real-time

## Technical details

- **Backend:** FastAPI + TensorFlow 2.15 + scikit-learn
- **Frontend:** Single-page HTML with MediaPipe Tasks JS (client-side)
- **Models:** 3 independent LSTMs (movement, audio, face) fused via confidence heuristic
- **Audio:** 22050 Hz, MFCC + delta features, SelectKBest feature selection

## Important note

This Space uses ephemeral storage. User data (recordings and trained models)
will be lost when the Space restarts. For persistent deployment, mount a
persistent storage volume at `/app/data`.
