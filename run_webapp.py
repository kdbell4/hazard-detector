"""
Entry point for the Hazard Detection web dashboard.

Usage:
    python run_webapp.py                       # webcam (default)
    python run_webapp.py --source 1            # external webcam
    python run_webapp.py --source video.mp4    # video file

Then open http://localhost:8000 in your browser.
"""

import os

# Must be set before cv2, pygame, or torch are imported anywhere.
# cv2 and pygame both bundle SDL2 — without this they fight over the
# display on macOS and cause a segfault.
os.environ["SDL_VIDEODRIVER"]            = "dummy"
os.environ["SDL_AUDIODRIVER"]            = "coreaudio"
# Fall back to CPU for any MPS ops not supported on this chip.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import uvicorn

parser = argparse.ArgumentParser(description="Hazard Detector Web Dashboard")
parser.add_argument(
    "--source",
    default="0",
    help="Camera ID (0, 1, …) or path to a video file (e.g. test.mp4). Default: 0",
)
args = parser.parse_args()

# Pass the source to the engine via environment variable
os.environ["HAZARD_SOURCE"] = args.source

if __name__ == "__main__":
    uvicorn.run(
        "webapp.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
