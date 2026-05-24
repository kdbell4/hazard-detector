# Hazard Detector

Real-time hazard detection for electric scooters using computer vision. A camera mounted on the scooter feeds live video into a pipeline that detects obstacles in the rider's path and triggers an audio alert before they become a collision risk.

---

## How it works

### Detection pipeline

Each frame from the webcam goes through up to six YOLO models running in parallel. Every detected object is then evaluated against three conditions — all three must be true for an alert to fire:

```
1. Too close?   — the object's bounding box covers enough of the frame
                  (>30% for people, >8% for everything else)

2. In the path? — the object's center falls within the middle 50% of
                  the frame horizontally (i.e. directly ahead)

3. Low enough?  — the bottom of the bounding box is in the lower 40%
                  of the frame, meaning it's on the ground near the scooter
```

If all three conditions are true, the object is flagged as a **HAZARD**, drawn in red, and an audio alert plays.

This three-condition approach avoids false alarms — a person far away in the distance passes condition 1 but not condition 3. A car parked to the side passes conditions 2 and 3 but not condition 2. Only genuine close-range obstacles in the scooter's direct path trigger an alert.

### Path detection (optional — with SAM)

With `mobile_sam.pt` installed, the app replaces the fixed horizontal center check with an actual road segmentation. The SAM model identifies the road surface in front of the scooter and only flags objects whose base point falls within that polygon. This is more accurate on curved paths or when the camera is angled.

Without SAM, the app falls back to the center-of-frame heuristic, which works well in most straight-line scenarios.

### Models

Six YOLO models cover different hazard categories:

| Model | Detects | Color |
|-------|---------|-------|
| Base (YOLOv8n) | People, cars, bikes, animals | 🟢 Green |
| Pothole (`best.pt`) | Potholes | 🔴 Red |
| Rocks (`rocks_best.pt`) | Rocks / debris | 🟠 Orange |
| Stairs (`stairs_best.pt`) | Stairs / steps | 🔵 Blue |
| Curb (`curb_best.pt`) | Curbs | 🟡 Yellow |
| Speed bump (`speedbump_best.pt`) | Speed bumps | 🟣 Magenta |

The custom models (pothole, rocks, stairs, curb, speedbump) were fine-tuned on domain-specific datasets. The base model uses the pretrained YOLOv8 nano weights from Ultralytics.

### Architecture

```
Webcam
  │
  ▼
CameraStream thread       ← continuously captures frames in background
  │
  ▼
DetectionEngine thread    ← owns all ML inference
  ├── SAM (every 5 frames)     → updates road path polygon
  └── 6× YOLO (every frame)    → run in parallel via ThreadPoolExecutor
        │
        ▼
  Hazard evaluation        ← area ratio + path check + vertical position
        │
        ├── Audio alert (pygame)
        ├── CSV log (hazard_log.csv)
        └── SharedState  ←──────────────────────┐
                                                 │
FastAPI server (main thread)                     │
  ├── GET  /              → web dashboard        │
  ├── GET  /video_feed    → MJPEG stream         │
  ├── GET  /api/status    → current hazards ─────┘
  ├── GET  /api/log       → detection history
  ├── POST /api/toggle_model  → enable/disable models at runtime
  └── POST /api/toggle_audio  → mute/unmute alerts
```

The `DetectionEngine` runs in a daemon thread and writes results to `SharedState`, a thread-safe data store. The FastAPI server reads from `SharedState` to serve the dashboard and API without blocking inference.

---

## Project structure

```
hazard-detector/
├── run_webapp.py              — launch the web dashboard
├── alert.mp3                  — audio played on hazard detection
├── models/                    — custom trained YOLO weights
│   ├── best.pt                  (pothole)
│   ├── rocks_best.pt            (rocks)
│   ├── stairs_best.pt           (stairs)
│   ├── curb_best.pt             (curbs)
│   ├── speedbump_best.pt        (speed bumps)
│   └── README.md
├── scripts/                   — standalone scripts (experiments / prototypes)
│   ├── detect.py                basic YOLO detection, no hazard logic
│   ├── distance.py              proximity-based hazard detection
│   ├── models.py                multi-model parallel detection
│   ├── yolo_alert.py            detection + audio alert
│   ├── yolo_alertlog.py         detection + audio + CSV logging
│   ├── path_detection.py        SAM road segmentation demo
│   └── test_cam.py              webcam sanity check
├── src/hazard_detector/       — installable Python package
│   ├── __main__.py              CLI entry point
│   └── detector.py              YOLODetector class
├── webapp/                    — web dashboard backend
│   ├── engine.py                DetectionEngine (inference thread)
│   ├── server.py                FastAPI routes
│   ├── state.py                 SharedState (thread-safe data store)
│   └── static/index.html        dashboard UI
└── pyproject.toml
```

---

## Requirements

- Python 3.8+
- A webcam (built-in or USB)
- **Recommended:** Mac with Apple Silicon (M1/M2/M3) — uses the Metal GPU automatically for 3–5× faster inference

---

## Install

```bash
git clone https://github.com/kdbell4/hazard-detector
cd hazard-detector
pip install -e .
```

Or without cloning (single scripts only):

```bash
pip install ultralytics opencv-python pygame
```

---

## Run

### Option 1 — Web dashboard (recommended)

```bash
python run_webapp.py
```

Open **http://localhost:8000** in any browser. The dashboard shows:
- Live annotated video feed
- Active hazards with object type, size, and which model detected it
- Per-model on/off toggles (disable models you don't need for better FPS)
- Audio mute toggle
- Detection log with timestamps

Anyone on the same Wi-Fi can view the stream at `http://<your-ip>:8000` — no install needed on their end.

### Option 2 — Webcam window

```bash
python -m hazard_detector webcam
```

Opens an OpenCV window. Press `q` to quit.

```bash
# If camera isn't found, try index 1
python -m hazard_detector webcam --camera-id 1
```

### Option 3 — Video file (no webcam needed)

```bash
python -m hazard_detector path/to/video.mp4
```

Works with any `.mp4`, `.mov`, or other video file. Good for testing without hardware.

**Free test footage:**
```bash
pip install yt-dlp
yt-dlp -f mp4 "https://www.youtube.com/watch?v=<video-id>" -o test_video.mp4
python -m hazard_detector test_video.mp4
```

---

## Performance

| Hardware | Expected FPS |
|----------|-------------|
| Mac Apple Silicon (M1/M2/M3) | 15–25 FPS |
| Mac Intel / Windows CPU | 4–10 FPS |
| NVIDIA GPU | 20–40 FPS |

The app auto-detects your hardware and uses MPS (Apple) or CUDA (NVIDIA) automatically. To improve speed, disable unused models in the web dashboard — each disabled model is one fewer parallel inference call per frame.

---

## Optional: SAM path detection

SAM (Segment Anything Model) makes hazard detection smarter by identifying the actual road surface rather than using a fixed center-of-frame zone.

1. Download `mobile_sam.pt` from the [MobileSAM releases page](https://github.com/ChaoningZhang/MobileSAM/releases)
2. Place it in the `models/` folder
3. Restart the app — SAM will be detected and enabled automatically

Without it the app works fine using the center-of-frame fallback.

---

## Troubleshooting

**Camera not opening:**
```bash
python -m hazard_detector webcam --camera-id 0
python -m hazard_detector webcam --camera-id 1
```

**Missing model weights:**
The app prints a warning and skips any missing model file rather than crashing. `yolov8n.pt` downloads automatically on first run. All custom `.pt` files are included in the repo under `models/`.

**pygame audio error:**
```bash
pip install pygame
```
