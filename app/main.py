import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# MediaPipe is not thread-safe — one worker ensures sequential processing
# while freeing the async event loop between frames.
_frame_executor = ThreadPoolExecutor(max_workers=1)

from .modules.camera import processor
from .modules.recorder import RecordingSession, DATA_PATH, ensure_csv
from .modules.trainer import TrainingSession, MODEL_PATH
from .modules.recognizer import recognizer
from .modules.audio_recorder import AudioRecordingSession, DATA_PATH as AUDIO_DATA_PATH, ensure_csv as audio_ensure_csv
from .modules.audio_trainer import AudioTrainingSession, MODEL_PATH as AUDIO_MODEL_PATH
from .modules.audio_recognizer import audio_recognizer

import app.modules.recorder as recorder_module
import app.modules.trainer as trainer_module
import app.modules.audio_recorder as audio_recorder_module
import app.modules.audio_trainer as audio_trainer_module

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

    legacy_gestures = BASE_DATA / "gestures.csv"
    if legacy_gestures.exists() and legacy_gestures.stat().st_size > 10:
        if not p["gestures_csv"].exists() or p["gestures_csv"].stat().st_size < 10:
            shutil.copy2(legacy_gestures, p["gestures_csv"])
            copied.append("gestures.csv")

    legacy_audio = BASE_DATA / "audio_gestures.csv"
    if legacy_audio.exists() and legacy_audio.stat().st_size > 10:
        if not p["audio_csv"].exists() or p["audio_csv"].stat().st_size < 10:
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


# ─── WebSocket: camera frames ─────────────────────────────────────────────────

@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            raw_jpeg = await websocket.receive_bytes()
            landmark_json = await loop.run_in_executor(_frame_executor, processor.process, raw_jpeg)
            await websocket.send_bytes(landmark_json)
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
        else:
            recognizer.load()
        processor.mode = "inference"
        processor.landmark_callback = recognizer.add_frame
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


# ─── Audio Inference ──────────────────────────────────────────────────────────

@app.post("/api/audio/inference/start")
async def start_audio_inference(username: str = Query(default="")):
    try:
        if username:
            _validate_username(username)
            p = _user_paths(username)
            audio_recognizer.load(model_path=p["au_model"], encoder_path=p["au_encoder"])
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


# ─── Fusion ───────────────────────────────────────────────────────────────────

@app.get("/api/fusion/prediction")
async def get_fusion_prediction():
    """Return movement prediction, audio prediction, and a fused result."""
    mv = recognizer.get_prediction()
    au = audio_recognizer.get_prediction()

    # Confidence-weighted fusion
    mv_conf = mv["confidence"] if mv["class"] != "—" else 0.0
    au_conf = au["confidence"] if au["class"] != "—" else 0.0
    total   = mv_conf + au_conf

    if total == 0:
        fused = {"class": "—", "confidence": 0.0}
    elif mv["class"] == au["class"]:
        fused = {"class": mv["class"], "confidence": min(1.0, total / 2 + 0.1)}
    elif mv_conf >= au_conf:
        fused = {"class": mv["class"], "confidence": mv_conf * 0.7}
    else:
        fused = {"class": au["class"], "confidence": au_conf * 0.7}

    return {"movement": mv, "audio": au, "fused": fused}
