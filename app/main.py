import json
import os
import re
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from .modules.camera import processor
from .modules.features import extract_landmarks_from_json
from .modules.recorder import RecordingSession, DATA_PATH, ensure_csv
from .modules.trainer import TrainingSession, MODEL_PATH
from .modules.recognizer import recognizer
from .modules.audio_recorder import AudioRecordingSession, DATA_PATH as AUDIO_DATA_PATH, ensure_csv as audio_ensure_csv
from .modules.audio_trainer import AudioTrainingSession, MODEL_PATH as AUDIO_MODEL_PATH
from .modules.audio_recognizer import audio_recognizer
from .modules.face_trainer import FaceTrainingSession
from .modules.face_recognizer import face_recognizer

import app.modules.recorder as recorder_module
import app.modules.trainer as trainer_module
import app.modules.audio_recorder as audio_recorder_module
import app.modules.audio_trainer as audio_trainer_module
import app.modules.face_trainer as face_trainer_module
import app.config as cfg

app = FastAPI(title="GestureAI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

BASE_DATA = Path(__file__).parent.parent / "data"

# ─── User helpers ─────────────────────────────────────────────────────────────

_USERNAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,50}$')

def _validate_username(username: str) -> str:
    if not username or not _USERNAME_RE.match(username):
        raise HTTPException(
            400,
            "Invalid username. Use letters, numbers, hyphens or underscores (max 50 chars)."
        )
    return username

def _user_dir(username: str) -> Path:
    return BASE_DATA / "users" / username

def _user_paths(username: str) -> dict:
    d = _user_dir(username)
    return {
        "data_dir":      d,
        "models_dir":    d / "models",
        "gestures_csv":  d / "gestures.csv",
        "audio_csv":     d / "audio_gestures.csv",
        "mv_model":      d / "models" / "movement_model.h5",
        "mv_encoder":    d / "models" / "label_encoder.pkl",
        "au_model":      d / "models" / "audio_model.h5",
        "au_encoder":    d / "models" / "audio_label_encoder.pkl",
        "face_model":    d / "models" / "face_model.h5",
        "face_encoder":  d / "models" / "face_label_encoder.pkl",
        "au_scaler":     d / "models" / "audio_scaler.pkl",
        "au_selector":   d / "models" / "audio_selector.pkl",
    }


# ─── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    processor.start()

@app.on_event("shutdown")
async def shutdown():
    processor.stop()


# ─── UI ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─── User management ──────────────────────────────────────────────────────────

class UserRequest(BaseModel):
    username: str

@app.get("/api/users")
async def list_users():
    users_dir = BASE_DATA / "users"
    if not users_dir.exists():
        return {"users": []}
    users = sorted(d.name for d in users_dir.iterdir() if d.is_dir())
    return {"users": users}

@app.post("/api/users")
async def create_user(req: UserRequest):
    username = _validate_username(req.username.strip())
    p = _user_paths(username)
    p["data_dir"].mkdir(parents=True, exist_ok=True)
    p["models_dir"].mkdir(parents=True, exist_ok=True)
    return {"username": username, "status": "ok"}


@app.post("/api/users/{username}/import-legacy")
async def import_legacy(username: str):
    """Copy legacy shared data/models into this user's folder (one-time migration)."""
    import shutil
    _validate_username(username)
    p = _user_paths(username)
    p["data_dir"].mkdir(parents=True, exist_ok=True)
    p["models_dir"].mkdir(parents=True, exist_ok=True)

    copied = []

    def _has_data(path: Path) -> bool:
        """Return True only if CSV has at least one data row (beyond the header)."""
        if not path.exists():
            return False
        with open(path) as f:
            f.readline()          # skip header
            return bool(f.readline().strip())   # True if there's a data row

    legacy_gestures = BASE_DATA / "gestures.csv"
    if _has_data(legacy_gestures) and not _has_data(p["gestures_csv"]):
        shutil.copy2(legacy_gestures, p["gestures_csv"])
        copied.append("gestures.csv")

    legacy_audio = BASE_DATA / "audio_gestures.csv"
    if _has_data(legacy_audio) and not _has_data(p["audio_csv"]):
        shutil.copy2(legacy_audio, p["audio_csv"])
        copied.append("audio_gestures.csv")

    legacy_models = {
        "movement_model.h5":      BASE_DATA / "models" / "movement_model.h5",
        "label_encoder.pkl":      BASE_DATA / "models" / "label_encoder.pkl",
        "audio_model.h5":         BASE_DATA / "models" / "audio_model.h5",
        "audio_label_encoder.pkl": BASE_DATA / "models" / "audio_label_encoder.pkl",
    }
    for fname, src in legacy_models.items():
        dst = p["models_dir"] / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(f"models/{fname}")

    if not copied:
        return {"status": "nothing_to_import", "message": "No legacy data found, or user data already exists."}
    return {"status": "ok", "copied": copied}


# ─── WebSocket: landmarks from browser ───────────────────────────────────────

@app.websocket("/ws/landmarks")
async def landmarks_ws(websocket: WebSocket):
    """Receive landmark JSON from browser-side MediaPipe Tasks JS."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            lm = extract_landmarks_from_json(data)
            if lm is not None and processor.landmark_callback:
                processor.landmark_callback(lm)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─── Data ─────────────────────────────────────────────────────────────────────

@app.get("/api/data/stats")
async def data_stats(username: str = Query(default="")):
    if username:
        _validate_username(username)
        p = _user_paths(username)
        csv_path   = p["gestures_csv"]
        model_path = p["mv_model"]
        ensure_csv(csv_path)
    else:
        csv_path   = DATA_PATH
        model_path = MODEL_PATH
        ensure_csv()

    if not csv_path.exists() or csv_path.stat().st_size < 10:
        return {"total_rows": 0, "classes": {}, "model_trained": False}
    try:
        df = pd.read_csv(csv_path)
        class_counts = df.groupby("class")["sequence_id"].nunique().to_dict()
        return {
            "total_rows": len(df),
            "classes": class_counts,
            "model_trained": model_path.exists(),
        }
    except Exception as e:
        return {"total_rows": 0, "classes": {}, "model_trained": False, "error": str(e)}


@app.delete("/api/data/gesture/{gesture_name}")
async def delete_gesture(gesture_name: str, username: str = Query(default="")):
    if username:
        _validate_username(username)
        csv_path = _user_paths(username)["gestures_csv"]
    else:
        csv_path = DATA_PATH

    if not csv_path.exists():
        raise HTTPException(404, "No data file found")
    df = pd.read_csv(csv_path)
    df = df[df["class"] != gesture_name]
    df.to_csv(csv_path, index=False)
    return {"status": "deleted", "gesture": gesture_name}


# ─── Recording ────────────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    gesture_name: str
    duration: int = 60
    username: str = ""


@app.post("/api/record/start")
async def start_recording(req: RecordRequest):
    if not req.gesture_name.strip():
        raise HTTPException(400, "Gesture name required")

    data_path = None
    if req.username:
        _validate_username(req.username)
        data_path = _user_paths(req.username)["gestures_csv"]

    session = RecordingSession(req.gesture_name.strip(), req.duration, data_path=data_path)
    recorder_module.current_session = session
    processor.landmark_callback = session.add_landmark
    processor.mode = "recording"
    session.start()
    return {"status": "recording", "gesture": req.gesture_name, "duration": req.duration}


@app.post("/api/record/stop")
async def stop_recording():
    session = recorder_module.current_session
    if session and session.status == "recording":
        session.stop()
    processor.landmark_callback = None
    processor.mode = "preview"
    return {"status": "stopped"}


@app.get("/api/record/status")
async def record_status():
    session = recorder_module.current_session
    if not session:
        return {"status": "idle", "progress": 0, "message": ""}
    return {
        "status": session.status,
        "progress": session.progress,
        "message": session.message,
        "frames_saved": session.frames_saved,
        "gesture": session.gesture_name,
    }


# ─── Training ─────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = 50
    username: str = ""


@app.post("/api/train/start")
async def start_training(req: TrainRequest):
    if trainer_module.current_training and trainer_module.current_training.status == "running":
        raise HTTPException(409, "Training already in progress")

    data_path = None
    model_dir = None
    if req.username:
        _validate_username(req.username)
        p = _user_paths(req.username)
        data_path = p["gestures_csv"]
        model_dir = p["models_dir"]

    session = TrainingSession(req.epochs, data_path=data_path, model_dir=model_dir)
    trainer_module.current_training = session
    session.start()
    return {"status": "started"}


@app.get("/api/train/status")
async def train_status():
    session = trainer_module.current_training
    if not session:
        return {"status": "idle", "progress": 0, "logs": [], "history": {}, "message": ""}
    return {
        "status": session.status,
        "progress": session.progress,
        "logs": session.logs[-20:],
        "history": session.history,
        "final_accuracy": session.final_accuracy,
        "message": session.message,
    }


# ─── Inference ────────────────────────────────────────────────────────────────

@app.post("/api/inference/start")
async def start_inference(username: str = Query(default="")):
    try:
        if username:
            _validate_username(username)
            p = _user_paths(username)
            recognizer.load(model_path=p["mv_model"], encoder_path=p["mv_encoder"])
            if p["face_model"].exists():
                face_recognizer.load(model_path=p["face_model"], encoder_path=p["face_encoder"])
        else:
            recognizer.load()
            fa_path = Path(__file__).parent.parent / "data" / "models" / "face_model.h5"
            if fa_path.exists():
                face_recognizer.load()

        def _combined_callback(lm):
            recognizer.add_frame(lm)
            if face_recognizer.is_loaded():
                face_recognizer.add_frame(lm)

        processor.mode = "inference"
        processor.landmark_callback = _combined_callback
        return {"status": "started"}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/inference/stop")
async def stop_inference():
    processor.mode = "preview"
    processor.landmark_callback = None
    return {"status": "stopped"}


@app.get("/api/inference/prediction")
async def get_prediction():
    pred = recognizer.get_prediction()
    if processor.mode == "inference":
        conf_pct = int(pred["confidence"] * 100)
        processor.prediction_overlay = f"{pred['class']} ({conf_pct}%)"
    return pred


# ─── WebSocket: browser mic audio ────────────────────────────────────────────

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_bytes()
            chunk = np.frombuffer(raw, dtype=np.float32)

            # Route to recording session
            session = audio_recorder_module.current_session
            if session and session.status == "recording":
                session.add_audio_chunk(chunk)

            # Route to recognizer (inference mode)
            if audio_recognizer.is_active():
                audio_recognizer.add_audio_chunk(chunk)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─── Audio Data ───────────────────────────────────────────────────────────────

@app.get("/api/audio/data/stats")
async def audio_data_stats(username: str = Query(default="")):
    if username:
        _validate_username(username)
        p = _user_paths(username)
        csv_path   = p["audio_csv"]
        model_path = p["au_model"]
        audio_ensure_csv(csv_path)
    else:
        csv_path   = AUDIO_DATA_PATH
        model_path = AUDIO_MODEL_PATH
        audio_ensure_csv()

    if not csv_path.exists() or csv_path.stat().st_size < 10:
        return {"total_rows": 0, "classes": {}, "model_trained": False}
    try:
        df = pd.read_csv(csv_path)
        class_counts = df.groupby("class")["sequence_id"].nunique().to_dict()
        return {
            "total_rows": len(df),
            "classes": class_counts,
            "model_trained": model_path.exists(),
        }
    except Exception as e:
        return {"total_rows": 0, "classes": {}, "model_trained": False, "error": str(e)}


@app.delete("/api/audio/data/gesture/{gesture_name}")
async def delete_audio_gesture(gesture_name: str, username: str = Query(default="")):
    if username:
        _validate_username(username)
        csv_path = _user_paths(username)["audio_csv"]
    else:
        csv_path = AUDIO_DATA_PATH

    if not csv_path.exists():
        raise HTTPException(404, "No audio data file found")
    df = pd.read_csv(csv_path)
    df = df[df["class"] != gesture_name]
    df.to_csv(csv_path, index=False)
    return {"status": "deleted", "gesture": gesture_name}


# ─── Audio Recording ──────────────────────────────────────────────────────────

class AudioRecordRequest(BaseModel):
    gesture_name: str
    duration: int = 10
    username: str = ""


@app.post("/api/audio/record/start")
async def start_audio_recording(req: AudioRecordRequest):
    if not req.gesture_name.strip():
        raise HTTPException(400, "Gesture name required")

    data_path = None
    if req.username:
        _validate_username(req.username)
        data_path = _user_paths(req.username)["audio_csv"]

    session = AudioRecordingSession(req.gesture_name.strip(), req.duration, data_path=data_path)
    audio_recorder_module.current_session = session
    session.start()
    return {"status": "recording", "gesture": req.gesture_name, "duration": req.duration}


@app.post("/api/audio/record/stop")
async def stop_audio_recording():
    session = audio_recorder_module.current_session
    if session and session.status == "recording":
        session.stop()
    return {"status": "stopped"}


@app.get("/api/audio/record/status")
async def audio_record_status():
    session = audio_recorder_module.current_session
    if not session:
        return {"status": "idle", "progress": 0, "message": ""}
    return {
        "status": session.status,
        "progress": session.progress,
        "message": session.message,
        "frames_saved": session.frames_saved,
        "gesture": session.gesture_name,
    }


# ─── Joint Recording ──────────────────────────────────────────────────────────

class JointRecordRequest(BaseModel):
    gesture_name: str
    duration: int = 15
    username: str = ""


@app.post("/api/joint/record/start")
async def start_joint_recording(req: JointRecordRequest):
    if not req.gesture_name.strip():
        raise HTTPException(400, "Gesture name required")

    gesture = req.gesture_name.strip()
    mv_data_path = None
    au_data_path = None
    if req.username:
        _validate_username(req.username)
        paths = _user_paths(req.username)
        mv_data_path = paths["gestures_csv"]
        au_data_path = paths["audio_csv"]

    mv_session = RecordingSession(gesture, req.duration, data_path=mv_data_path)
    recorder_module.current_session = mv_session
    processor.landmark_callback = mv_session.add_landmark
    processor.mode = "recording"
    mv_session.start()

    au_session = AudioRecordingSession(gesture, req.duration, data_path=au_data_path)
    audio_recorder_module.current_session = au_session
    au_session.start()

    return {"status": "recording", "gesture": gesture, "duration": req.duration}


@app.post("/api/joint/record/stop")
async def stop_joint_recording():
    mv_session = recorder_module.current_session
    if mv_session and mv_session.status == "recording":
        mv_session.stop()
    processor.landmark_callback = None
    processor.mode = "preview"

    au_session = audio_recorder_module.current_session
    if au_session and au_session.status == "recording":
        au_session.stop()
    return {"status": "stopped"}


@app.get("/api/joint/record/status")
async def joint_record_status():
    mv = recorder_module.current_session
    au = audio_recorder_module.current_session
    mv_s = {"status": "idle", "progress": 0.0, "frames_saved": 0} if not mv else {
        "status": mv.status, "progress": mv.progress, "frames_saved": mv.frames_saved
    }
    au_s = {"status": "idle", "progress": 0.0, "frames_saved": 0} if not au else {
        "status": au.status, "progress": au.progress, "frames_saved": au.frames_saved
    }
    combined = "idle"
    if mv_s["status"] == "recording" or au_s["status"] == "recording":
        combined = "recording"
    elif mv_s["status"] == "error" or au_s["status"] == "error":
        combined = "error"
    elif mv_s["status"] == "done" or au_s["status"] == "done":
        combined = "done"
    return {
        "status": combined,
        "progress": (mv_s["progress"] + au_s["progress"]) / 2,
        "movement": mv_s,
        "audio": au_s,
    }


# ─── Audio Training ───────────────────────────────────────────────────────────

class AudioTrainRequest(BaseModel):
    epochs: int = 50
    username: str = ""


@app.post("/api/audio/train/start")
async def start_audio_training(req: AudioTrainRequest):
    if audio_trainer_module.current_training and audio_trainer_module.current_training.status == "running":
        raise HTTPException(409, "Audio training already in progress")

    data_path = None
    model_dir = None
    if req.username:
        _validate_username(req.username)
        p = _user_paths(req.username)
        data_path = p["audio_csv"]
        model_dir = p["models_dir"]

    session = AudioTrainingSession(req.epochs, data_path=data_path, model_dir=model_dir)
    audio_trainer_module.current_training = session
    session.start()
    return {"status": "started"}


@app.get("/api/audio/train/status")
async def audio_train_status():
    session = audio_trainer_module.current_training
    if not session:
        return {"status": "idle", "progress": 0, "logs": [], "history": {}, "message": ""}
    return {
        "status": session.status,
        "progress": session.progress,
        "logs": session.logs[-20:],
        "history": session.history,
        "final_accuracy": session.final_accuracy,
        "message": session.message,
    }


# ─── Face Training ────────────────────────────────────────────────────────────

class FaceTrainRequest(BaseModel):
    epochs: int = 50
    username: str = ""


@app.post("/api/face/train/start")
async def start_face_training(req: FaceTrainRequest):
    if face_trainer_module.current_training and face_trainer_module.current_training.status == "running":
        raise HTTPException(409, "Face training already in progress")

    data_path = None
    model_dir = None
    if req.username:
        _validate_username(req.username)
        p = _user_paths(req.username)
        data_path = p["gestures_csv"]
        model_dir = p["models_dir"]

    session = FaceTrainingSession(req.epochs, data_path=data_path, model_dir=model_dir)
    face_trainer_module.current_training = session
    session.start()
    return {"status": "started"}


@app.get("/api/face/train/status")
async def face_train_status():
    session = face_trainer_module.current_training
    if not session:
        return {"status": "idle", "progress": 0, "logs": [], "history": {}, "message": ""}
    return {
        "status": session.status,
        "progress": session.progress,
        "logs": session.logs[-20:],
        "history": session.history,
        "final_accuracy": session.final_accuracy,
        "message": session.message,
    }




# ─── Audio Inference ──────────────────────────────────────────────────────────

@app.post("/api/audio/inference/start")
async def start_audio_inference(username: str = Query(default="")):
    try:
        if username:
            _validate_username(username)
            p = _user_paths(username)
            audio_recognizer.load(
                model_path=p["au_model"], encoder_path=p["au_encoder"],
                scaler_path=p["au_scaler"], selector_path=p["au_selector"],
            )
        else:
            audio_recognizer.load()
        audio_recognizer.start()
        return {"status": "started"}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/audio/inference/stop")
async def stop_audio_inference():
    audio_recognizer.stop()
    return {"status": "stopped"}


@app.get("/api/audio/inference/prediction")
async def get_audio_prediction():
    return audio_recognizer.get_prediction()


# ─── Video file inference ─────────────────────────────────────────────────────

def _analyse_video(video_path: str, model_path: Path, encoder_path: Path) -> dict:
    """Read a video file frame-by-frame, run MediaPipe + LSTM, return predictions."""
    import cv2
    import joblib
    import mediapipe as mp
    from collections import deque
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from .modules.features import extract_landmarks

    ANGLE_COLS = np.concatenate([np.arange(162, 176), np.arange(239, 253)])

    model = load_model(model_path)
    le    = joblib.load(encoder_path)
    try:
        seq_len = model.input_shape[1] or 11
    except Exception:
        seq_len = 11

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    frame_skip   = max(1, int(fps / 15))   # sample at ~15 fps

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window      = deque(maxlen=seq_len)
    predictions = []
    frame_idx   = 0
    last_pred_t = -1.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            t   = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            lm = extract_landmarks(results)
            if lm is not None:
                window.append(lm)
                if len(window) >= seq_len and (t - last_pred_t) >= 0.3:
                    last_pred_t = t
                    seq = np.array(list(window), dtype="float32")
                    seq[:, ANGLE_COLS] /= 180.0
                    X     = pad_sequences([seq], padding="post", dtype="float32", value=-1.0)
                    probs = model.predict(X, verbose=0)[0]
                    idx   = int(np.argmax(probs))
                    conf  = float(probs[idx])
                    if conf >= cfg.CONFIDENCE_THRESHOLD:
                        predictions.append({
                            "time":       round(t, 2),
                            "class":      le.classes_[idx],
                            "confidence": round(conf, 3),
                        })
            frame_idx += 1
    finally:
        cap.release()
        holistic.close()

    summary = {}
    for p in predictions:
        summary[p["class"]] = summary.get(p["class"], 0) + 1

    dominant = max(summary, key=summary.get) if summary else "—"
    return {
        "predictions": predictions,
        "summary":     summary,
        "dominant":    dominant,
        "duration":    round(duration, 2),
        "frames_analysed": frame_idx // frame_skip,
    }


@app.post("/api/inference/video")
async def inference_video(
    file: UploadFile = File(...),
    username: str = Query(default=""),
):
    """Upload a video file and get gesture predictions from it."""
    if username:
        _validate_username(username)
        p = _user_paths(username)
        model_path, encoder_path = p["mv_model"], p["mv_encoder"]
    else:
        model_path, encoder_path = MODEL_PATH, Path(str(MODEL_PATH).replace("movement_model.h5", "label_encoder.pkl"))

    if not model_path.exists():
        raise HTTPException(404, "No trained model found. Train a movement model first.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _analyse_video, tmp_path, model_path, encoder_path
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


# ─── Video file training data extraction ─────────────────────────────────────

def _extract_video_training_data(video_path: str, gesture_name: str, data_path: Path) -> dict:
    """Run MediaPipe on a video, identify keyframes, save rows to gestures CSV."""
    import csv
    import cv2
    import mediapipe as mp
    import pandas as pd
    from .modules.features import extract_landmarks, identify_keyframes
    from .modules.recorder import ensure_csv

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(fps / 15))   # sample at ~15 fps

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            lm = extract_landmarks(results)
            if lm is not None:
                landmarks.append(lm)
            frame_idx += 1
    finally:
        cap.release()
        holistic.close()

    if len(landmarks) < 3:
        raise ValueError("Not enough usable frames detected. Ensure hands/body are visible throughout the video.")

    arr      = np.array(landmarks)
    keyframes = identify_keyframes(arr)
    if not keyframes:
        raise ValueError("No significant gesture keyframes detected in video.")

    ensure_csv(data_path)
    if data_path.exists() and data_path.stat().st_size > 0:
        df      = pd.read_csv(data_path)
        next_id = int(df["sequence_id"].max()) + 1 if len(df) > 0 else 1
    else:
        next_id = 1

    rows_written = 0
    with open(data_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        seq_id = next_id
        for kf in keyframes:
            start_idx = max(0, kf - cfg.FRAME_WINDOW)
            end_idx   = min(len(arr), kf + cfg.FRAME_WINDOW + 1)
            for idx in range(start_idx, end_idx):
                writer.writerow([gesture_name, seq_id] + list(arr[idx]))
                rows_written += 1
            seq_id += 1

    return {
        "status":    "done",
        "gesture":   gesture_name,
        "rows_saved": rows_written,
        "sequences": len(keyframes),
        "message":   f"Saved {rows_written} rows ({len(keyframes)} sequences) for '{gesture_name}'",
    }


@app.post("/api/record/video-file")
async def record_from_video(
    file:         UploadFile = File(...),
    gesture_name: str        = Query(...),
    username:     str        = Query(default=""),
):
    """Process a video file through MediaPipe and save extracted keyframes as training data."""
    if not gesture_name.strip():
        raise HTTPException(400, "Gesture name required")

    if username:
        _validate_username(username)
        data_path = _user_paths(username)["gestures_csv"]
    else:
        data_path = DATA_PATH

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _extract_video_training_data, tmp_path, gesture_name.strip(), data_path
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


# ─── Fusion ───────────────────────────────────────────────────────────────────
# Thresholds read from cfg at call time — live-updateable via /api/config/apply

def _heuristic_fusion(mv: dict, face: dict, au: dict, hand_active: bool = False) -> dict:
    mc, mf = mv["class"],   mv["confidence"]
    ac, af = au["class"],   au["confidence"]
    fc, ff = face["class"], face["confidence"]

    theta_aud     = cfg.THETA_AUD
    theta_vis     = cfg.THETA_VIS
    w_agree       = cfg.W_AGREE
    w_disagree    = cfg.W_DISAGREE

    # Audio override: strong audio signal wins regardless of movement
    if ac != "—" and af >= theta_aud:
        agreed = mc == ac
        conf = min(1.0, af * w_agree) if agreed else af
        modifier = f"Movement agreed → ×{w_agree}" if agreed else "Movement disagreed (no boost)"
        return {
            "class": ac, "confidence": round(conf, 3),
            "rule": "audio_override",
            "rule_label": f"Audio override — {round(af*100)}% ≥ {round(theta_aud*100)}%",
            "detail": modifier,
        }

    # Movement primary: above vision threshold
    if mc != "—" and mf >= theta_vis:
        conf = mf
        modifiers = []
        if ac == mc:
            conf = min(1.0, conf * w_agree)
            modifiers.append(f"Audio agrees → ×{w_agree}")
        elif ac != "—":
            conf *= w_disagree
            modifiers.append(f"Audio disagrees → ×{w_disagree}")
        # Face agreement boost only when hands are NOT active
        if not hand_active and fc == mc:
            conf = min(1.0, conf * 1.05)
            modifiers.append("Face agrees → ×1.05")
        elif hand_active:
            modifiers.append("Face suppressed (hands active)")
        return {
            "class": mc, "confidence": round(conf, 3),
            "rule": "movement_primary",
            "rule_label": f"Movement primary — {round(mf*100)}% ≥ {round(theta_vis*100)}%",
            "detail": ", ".join(modifiers) if modifiers else "No modifiers applied",
        }

    # Below all thresholds: best of movement/audio wins (face is modifier only).
    candidates = [(mc, mf, "movement"), (ac, af, "audio")]
    candidates = [(c, f, src) for c, f, src in candidates if c and c != "—"]
    if not candidates:
        return {
            "class": "—", "confidence": 0.0,
            "rule": "no_signal",
            "rule_label": "No gesture detected",
            "detail": f"All streams below thresholds (Movement<{round(theta_vis*100)}%, Audio<{round(theta_aud*100)}%)",
        }
    best_c, best_f, best_src = max(candidates, key=lambda x: x[1])

    # THETA_MIN gate: if best candidate is below minimum, report no gesture
    theta_min = cfg.THETA_MIN
    if best_f < theta_min:
        return {
            "class": "—", "confidence": 0.0,
            "rule": "below_minimum",
            "rule_label": "No gesture detected",
            "detail": f"Best stream ({best_src}) {round(best_f*100)}% < {round(theta_min*100)}% minimum",
        }

    conf = best_f * w_disagree
    modifiers = []
    # Face acts as supporting modifier only — never primary
    if fc == best_c and not hand_active:
        conf = min(1.0, conf * 1.05)
        modifiers.append("Face agrees → ×1.05")
    return {
        "class": best_c, "confidence": round(conf, 3),
        "rule": "low_confidence",
        "rule_label": f"Low confidence — best stream {round(best_f*100)}% (discounted)",
        "detail": f"Movement {round(mf*100)}%, Audio {round(af*100)}% — winner: {best_src}" +
                  (f", {', '.join(modifiers)}" if modifiers else ""),
    }


@app.get("/api/fusion/prediction")
async def get_fusion_prediction():
    """Return movement, face, audio predictions plus confidence-weighted fused result."""
    mv   = recognizer.get_prediction()
    face = face_recognizer.get_prediction() if face_recognizer.is_loaded() else {"class": "—", "confidence": 0.0}
    au   = audio_recognizer.get_prediction()
    hand_active = recognizer.hands_active()
    fused = _heuristic_fusion(mv, face, au, hand_active=hand_active)
    return {"movement": mv, "face": face, "audio": au, "fused": fused, "hand_active": hand_active}


# ─── Config ───────────────────────────────────────────────────────────────────

# Keys that take effect immediately (read at call time, not cached at startup)
_LIVE_KEYS = {
    "THETA_AUD", "THETA_VIS", "W_AGREE", "W_DISAGREE",
    "CONFIDENCE_THRESHOLD",
    "COLOR_LEFT_HAND", "COLOR_RIGHT_HAND", "COLOR_POSE", "COLOR_FACE_OVAL",
}

# Keys that require a server restart to take effect (baked into buffers/models at init)
_RESTART_KEYS = {
    "SAMPLE_RATE", "WINDOW_DURATION", "HOP_DURATION", "N_MFCC",
    "FPS_CAP", "JPEG_QUALITY",
    "MEDIAPIPE_MODEL_COMPLEXITY", "MEDIAPIPE_DETECTION_CONFIDENCE", "MEDIAPIPE_TRACKING_CONFIDENCE",
    "WINDOW_SIZE", "PREDICTION_INTERVAL",
    "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "TEST_SIZE",
    "KBEST_K",
}


@app.get("/api/config")
async def get_config():
    all_keys = _LIVE_KEYS | _RESTART_KEYS
    return {
        "values": {k: getattr(cfg, k) for k in all_keys},
        "live_keys": sorted(_LIVE_KEYS),
        "restart_keys": sorted(_RESTART_KEYS),
    }


@app.post("/api/config/apply")
async def apply_config(updates: dict):
    applied, skipped = {}, {}
    for key, value in updates.items():
        if key in _LIVE_KEYS:
            setattr(cfg, key, type(getattr(cfg, key))(value))
            applied[key] = value
        elif key in _RESTART_KEYS:
            skipped[key] = "restart required"
        else:
            skipped[key] = "unknown key"
    return {"status": "applied", "applied": applied, "skipped": skipped}


@app.post("/api/config/reset")
async def reset_config():
    import importlib
    import app.config as _cfg_mod
    importlib.reload(_cfg_mod)
    # Re-point the local reference so the reloaded values are used
    global cfg
    import app.config as cfg  # noqa: F811
    return {"status": "reset"}


@app.post("/api/camera/reinitialise")
async def reinitialise_camera():
    """Restart MediaPipe Holistic (pick up updated model_complexity etc.)."""
    processor.stop()
    processor.start()
    return {"status": "reinitialised"}
