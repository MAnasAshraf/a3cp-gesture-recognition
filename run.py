#!/usr/bin/env python3
"""
A3CP – Multimodal Assistive Communication
Start the web server: python run.py
Then open http://localhost:8000 in your browser.
"""
import os

# Limit TensorFlow threading to prevent pthread_create failures on
# resource-constrained containers (Railway, Heroku, etc.)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
