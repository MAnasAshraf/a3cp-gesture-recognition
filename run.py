#!/usr/bin/env python3
"""
A3CP – Multimodal Assistive Communication

Desktop mode (default):
    python run.py               # opens app in a native PyWebView window

Browser mode:
    python run.py --no-window   # starts server and opens default browser

First-time offline setup (download CDN assets):
    python setup_offline.py
"""

import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def start_server() -> uvicorn.Server:
    config = uvicorn.Config(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server


def wait_for_server(timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/", timeout=1)
            return True
        except Exception:
            time.sleep(0.1)
    return False


if __name__ == "__main__":
    use_window = "--no-window" not in sys.argv

    start_server()

    if not wait_for_server():
        print("ERROR: Server failed to start within 10 seconds.")
        sys.exit(1)

    url = f"http://{HOST}:{PORT}/"

    if use_window:
        try:
            import webview  # type: ignore

            window = webview.create_window(
                "A3CP – Assistive Communication",
                url,
                width=1280,
                height=800,
                resizable=True,
            )
            webview.start()
            # webview.start() blocks until the window is closed;
            # the daemon server thread exits automatically with the process.
        except ImportError:
            print(
                "pywebview is not installed — opening browser instead.\n"
                "Install it with:  pip install pywebview\n"
                "  macOS/Windows: no extra system packages needed.\n"
                "  Linux: apt install python3-gi gir1.2-webkit2-4.0"
            )
            webbrowser.open(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    else:
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
