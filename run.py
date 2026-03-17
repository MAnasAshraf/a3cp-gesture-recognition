#!/usr/bin/env python3
"""
A3CP – Multimodal Assistive Communication
Start the web server: python run.py
Then open http://localhost:8000 in your browser.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
