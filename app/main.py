import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from .modules.camera import processor
from .modules.recorder import RecordingSession, DATA_PATH, ensure_csv
from .modules.trainer import TrainingSession, MODEL_PATH, ENCODER_PATH
from .modules.recognizer import recognizer

import app.modules.recorder as recorder_module
import app.modules.trainer as trainer_module

app = FastAPI(title="GestureAI")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


# ─── WebSocket: camera frames ─────────────────────────────────────────────────
# Browser sends raw JPEG bytes → backend runs MediaPipe → sends annotated JPEG back.
# This avoids all macOS camera permission issues in the backend process.

@app.websocket("/ws/camera")
async def camera_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_jpeg = await websocket.receive_bytes()
            annotated_jpeg = processor.process(raw_jpeg)
            await websocket.send_bytes(annotated_jpeg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─── Data ─────────────────────────────────────────────────────────────────────

@app.get("/api/data/stats")
async def data_stats():
    ensure_csv()
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size < 10:
        return {"total_rows": 0, "classes": {}, "model_trained": False}
    try:
        df = pd.read_csv(DATA_PATH)
        class_counts = df.groupby("class")["sequence_id"].nunique().to_dict()
        return {
            "total_rows": len(df),
            "classes": class_counts,
            "model_trained": MODEL_PATH.exists(),
        }
    except Exception as e:
        return {"total_rows": 0, "classes": {}, "model_trained": False, "error": str(e)}


@app.delete("/api/data/gesture/{gesture_name}")
async def delete_gesture(gesture_name: str):
    if not DATA_PATH.exists():
        raise HTTPException(404, "No data file found")
    df = pd.read_csv(DATA_PATH)
    df = df[df["class"] != gesture_name]
    df.to_csv(DATA_PATH, index=False)
    return {"status": "deleted", "gesture": gesture_name}


# ─── Recording ────────────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    gesture_name: str
    duration: int = 60


@app.post("/api/record/start")
async def start_recording(req: RecordRequest):
    if not req.gesture_name.strip():
        raise HTTPException(400, "Gesture name required")
    session = RecordingSession(req.gesture_name.strip(), req.duration)
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


@app.post("/api/train/start")
async def start_training(req: TrainRequest):
    if trainer_module.current_training and trainer_module.current_training.status == "running":
        raise HTTPException(409, "Training already in progress")
    session = TrainingSession(req.epochs)
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
async def start_inference():
    try:
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
