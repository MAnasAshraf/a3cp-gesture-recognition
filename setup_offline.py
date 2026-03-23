#!/usr/bin/env python3
"""
A3CP – Offline Asset Setup
Run once to download all CDN dependencies for offline / desktop use.

Usage:
    python setup_offline.py          # download missing assets only
    python setup_offline.py --force  # re-download everything

After running this script, start the app with:
    python run.py
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
STATIC = ROOT / "app" / "static"
VENDOR = STATIC / "vendor"
FONTS_DIR = VENDOR / "fonts"
MEDIAPIPE_DIR = VENDOR / "mediapipe"
WASM_DIR = MEDIAPIPE_DIR / "wasm"
MODELS_DIR = MEDIAPIPE_DIR / "models"
INDEX_HTML = STATIC / "index.html"
MANIFEST_FILE = VENDOR / ".offline-manifest.json"

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
CHARTJS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
CHARTJS_DEST = VENDOR / "chart.umd.min.js"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap"
)
FONTS_CSS_DEST = FONTS_DIR / "poppins.css"

MEDIAPIPE_VERSION = "0.10.18"
MEDIAPIPE_BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MEDIAPIPE_VERSION}"
MEDIAPIPE_BUNDLE_URL = f"{MEDIAPIPE_BASE}/vision_bundle.mjs"
MEDIAPIPE_BUNDLE_DEST = MEDIAPIPE_DIR / "vision_bundle.mjs"

WASM_FILES = [
    "vision_wasm_internal.js",
    "vision_wasm_internal.wasm",
    "vision_wasm_nosimd_internal.js",
    "vision_wasm_nosimd_internal.wasm",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_DEST = MODELS_DIR / "holistic_landmarker.task"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str, force: bool = False, show_progress: bool = False) -> int:
    """Download url to dest. Returns file size in bytes. Skips if dest exists and not forced."""
    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"  [skip] {label} ({size:,} bytes already present)")
        return size

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {label} …", end="", flush=True)

    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        data = b""
        chunk_size = 65536
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            data += chunk
            downloaded += len(chunk)
            if show_progress and total:
                pct = downloaded * 100 // total
                print(f"\r  [download] {label} … {pct}% ({downloaded:,}/{total:,} bytes)", end="", flush=True)

    dest.write_bytes(data)
    print(f"\r  [ok] {label} ({len(data):,} bytes)      ")
    return len(data)


def _patch_file(path: Path, old: str, new: str, description: str) -> bool:
    """Replace old with new in file. Returns True if a change was made."""
    content = path.read_text(encoding="utf-8")
    if old not in content:
        if new in content:
            print(f"  [skip] {description} (already patched)")
            return False
        print(f"  [warn] {description} – pattern not found, skipping")
        return False
    patched = content.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"  [patch] {description}")
    return True


# ---------------------------------------------------------------------------
# Step 1 – Google Fonts (Poppins)
# ---------------------------------------------------------------------------

def download_fonts(force: bool) -> dict:
    manifest = {}
    print("\n=== Step 1: Google Fonts (Poppins) ===")
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch the CSS (desktop UA so we get woff2 for all weights)
    req = urllib.request.Request(GOOGLE_FONTS_URL, headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })
    print("  [fetch] Google Fonts CSS …", end="", flush=True)
    with urllib.request.urlopen(req, timeout=30) as resp:
        css_text = resp.read().decode("utf-8")
    print(" ok")

    # Extract all woff2 URLs from the CSS
    woff2_urls = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+\.woff2)\)", css_text)
    if not woff2_urls:
        print("  [warn] No woff2 URLs found in Google Fonts CSS – check User-Agent")

    # Build local filenames from the URL paths
    local_css = css_text
    for url in woff2_urls:
        # Use the last path component as the filename (it's a hash-based name)
        fname = url.split("/")[-1]
        # Make a friendlier name based on the @font-face context (weight)
        # We'll keep the original hash name for uniqueness
        dest = FONTS_DIR / fname
        size = _download(url, dest, f"font {fname}", force=force)
        manifest[f"fonts/{fname}"] = {"url": url, "size": size}
        # Rewrite the CSS to use the local path
        local_css = local_css.replace(url, f"/static/vendor/fonts/{fname}")

    # Write the local CSS
    FONTS_CSS_DEST.write_text(local_css, encoding="utf-8")
    print(f"  [write] poppins.css ({len(local_css)} chars)")
    manifest["fonts/poppins.css"] = {"url": GOOGLE_FONTS_URL, "size": len(local_css)}
    return manifest


# ---------------------------------------------------------------------------
# Step 2 – Chart.js
# ---------------------------------------------------------------------------

def download_chartjs(force: bool) -> dict:
    print("\n=== Step 2: Chart.js ===")
    size = _download(CHARTJS_URL, CHARTJS_DEST, "chart.umd.min.js", force=force)
    return {"chart.umd.min.js": {"url": CHARTJS_URL, "size": size}}


# ---------------------------------------------------------------------------
# Step 3 – MediaPipe Tasks Vision
# ---------------------------------------------------------------------------

def download_mediapipe(force: bool) -> dict:
    print("\n=== Step 3: MediaPipe Tasks Vision ===")
    manifest = {}

    # ES module bundle
    size = _download(MEDIAPIPE_BUNDLE_URL, MEDIAPIPE_BUNDLE_DEST, "vision_bundle.mjs", force=force)
    manifest["mediapipe/vision_bundle.mjs"] = {"url": MEDIAPIPE_BUNDLE_URL, "size": size}

    # Inspect bundle for any hardcoded CDN references
    if MEDIAPIPE_BUNDLE_DEST.exists():
        bundle_text = MEDIAPIPE_BUNDLE_DEST.read_text(encoding="utf-8", errors="replace")
        cdn_refs = [m for m in re.findall(r"https://cdn\.jsdelivr\.net[^\s\"']+", bundle_text)
                    if "tasks-vision" in m]
        storage_refs = [m for m in re.findall(r"https://storage\.googleapis\.com[^\s\"']+", bundle_text)]
        if cdn_refs:
            print(f"  [info] Bundle contains {len(cdn_refs)} internal jsDelivr reference(s) – WASM will still load from local path via FilesetResolver")
        if storage_refs:
            print(f"  [warn] Bundle contains {len(storage_refs)} storage.googleapis.com reference(s) – model URL patching in index.html is required")

    # WASM files
    WASM_DIR.mkdir(parents=True, exist_ok=True)
    for wasm_file in WASM_FILES:
        url = f"{MEDIAPIPE_BASE}/wasm/{wasm_file}"
        dest = WASM_DIR / wasm_file
        size = _download(url, dest, f"wasm/{wasm_file}", force=force)
        manifest[f"mediapipe/wasm/{wasm_file}"] = {"url": url, "size": size}

    return manifest


# ---------------------------------------------------------------------------
# Step 4 – HolisticLandmarker model (~40 MB)
# ---------------------------------------------------------------------------

def download_model(force: bool) -> dict:
    print("\n=== Step 4: HolisticLandmarker model (~40 MB) ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    size = _download(MODEL_URL, MODEL_DEST, "holistic_landmarker.task", force=force, show_progress=True)
    return {"mediapipe/models/holistic_landmarker.task": {"url": MODEL_URL, "size": size}}


# ---------------------------------------------------------------------------
# Step 5 – Patch index.html
# ---------------------------------------------------------------------------

def patch_index_html() -> None:
    print("\n=== Step 5: Patching index.html ===")

    # 1. Google Fonts preconnect tags + font CSS link  →  local CSS link
    _patch_file(
        INDEX_HTML,
        old=(
            '  <link rel="preconnect" href="https://fonts.googleapis.com" />\n'
            '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n'
            '  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet" />'
        ),
        new='  <link href="/static/vendor/fonts/poppins.css" rel="stylesheet" />',
        description="Google Fonts → local poppins.css",
    )

    # 2. Chart.js CDN script  →  local script
    _patch_file(
        INDEX_HTML,
        old='  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>',
        new='  <script src="/static/vendor/chart.umd.min.js"></script>',
        description="Chart.js CDN → local",
    )

    # 3. MediaPipe ES module import URL  →  local bundle
    _patch_file(
        INDEX_HTML,
        old=(
            '    import { FilesetResolver, HolisticLandmarker } from\n'
            '      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";'
        ),
        new=(
            '    import { FilesetResolver, HolisticLandmarker } from\n'
            '      "/static/vendor/mediapipe/vision_bundle.mjs";'
        ),
        description="MediaPipe ES import → local bundle",
    )

    # 4. FilesetResolver WASM base URL  →  local wasm directory
    _patch_file(
        INDEX_HTML,
        old='    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"',
        new='    "/static/vendor/mediapipe/wasm"',
        description="FilesetResolver WASM path → local",
    )

    # 5. modelAssetPath (storage.googleapis.com)  →  local model
    _patch_file(
        INDEX_HTML,
        old=(
            '      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/'
            'holistic_landmarker/float16/latest/holistic_landmarker.task",'
        ),
        new='      modelAssetPath: "/static/vendor/mediapipe/models/holistic_landmarker.task",',
        description="modelAssetPath → local model",
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(manifest: dict) -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    data = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": manifest,
    }
    MANIFEST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  [manifest] Written to {MANIFEST_FILE.relative_to(ROOT)}")


def verify_no_cdn_refs() -> bool:
    print("\n=== Verification ===")
    content = INDEX_HTML.read_text(encoding="utf-8")
    patterns = ["fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "storage.googleapis.com"]
    found = [p for p in patterns if p in content]
    if found:
        print(f"  [FAIL] index.html still contains CDN references: {found}")
        return False
    print("  [ok] index.html contains no external CDN references")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download offline assets for A3CP")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    print("A3CP Offline Setup")
    print("=" * 50)
    if args.force:
        print("Mode: force re-download")
    else:
        print("Mode: skip existing files (use --force to re-download)")

    manifest = {}
    manifest.update(download_fonts(args.force))
    manifest.update(download_chartjs(args.force))
    manifest.update(download_mediapipe(args.force))
    manifest.update(download_model(args.force))
    patch_index_html()
    write_manifest(manifest)
    ok = verify_no_cdn_refs()

    print("\n" + "=" * 50)
    if ok:
        print("Setup complete. Run:  python run.py")
    else:
        print("Setup completed with warnings — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
