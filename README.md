# Hazard Detector

Real-time hazard detection for electric scooters using computer vision. A camera mounted on the scooter feeds live video into a pipeline that detects obstacles in the rider's path and triggers an audio alert before they become a collision risk.

---

## Team

Each part of the system was built by a different person and then integrated into the final app.

### 🟠 Katherine — Hazard Detection (model training & datasets)

Responsible for identifying which hazards to detect, sourcing training data, and producing the custom YOLO model weights in `models/`.

**What YOLOv8 already detects out of the box** (no custom training needed):

| Class | ID | Relevant to scooters? |
|-------|----|-----------------------|
| person | 0 | ✅ |
| bicycle | 1 | ✅ |
| car | 2 | ✅ |
| motorcycle | 3 | ✅ |
| bus | 5 | ✅ |
| truck | 7 | ✅ |
| traffic light | 9 | ✅ |
| fire hydrant | 10 | ✅ |
| stop sign | 11 | ✅ |
| bench | 13 | ✅ |
| bird | 14 | ✅ |
| dog | 16 | ✅ |
| cat | 15 | ✅ |
| skateboard | 36 | ✅ |

**Hazards that required custom-trained models** (not in COCO):

| Hazard | Model file | Public datasets used |
|--------|-----------|----------------------|
| Potholes | `models/best.pt` | [Road Damage Dataset – Roboflow](https://universe.roboflow.com/yolo-hwkmv/road-qrmur) |
| Rocks / debris | `models/rocks_best.pt` | [Rock Detection – Roboflow](https://universe.roboflow.com/yolo-olcxk/rock-detection-9kgmi-i51q4) · [Rock Instance Segmentation – Roboflow](https://universe.roboflow.com/instance-segmentation-pnlez/rock-detection-fmyg7) |
| Stairs | `models/stairs_best.pt` | Custom collected |
| Curbs | `models/curb_best.pt` | Custom collected |
| Speed bumps | `models/speedbump_best.pt` | Custom collected |

> **Puddles** are a planned addition. Existing datasets for training:
> - [Puddles Object Detection – Roboflow](https://universe.roboflow.com/water-yzjeu/puddles-7mvza-hwecr)
> - [Puddle Detection – Hanyang University / Roboflow](https://universe.roboflow.com/hanyang-university-bd2kb/puddle-detection) (~1,500 images, 79.6% mAP)

---

### 📏 Ashwin — Distance Estimation

Built the proximity logic that determines how close an object is to the scooter without a depth sensor, using bounding box area as a proxy for distance.

The key insight: the closer an object is, the larger its bounding box relative to the frame. The ratio of box area to total frame area is computed for every detection:

```python
area_ratio = (box_width * box_height) / (frame_width * frame_height)
```

Class-specific thresholds are used because people appear large even when far away:

```python
too_close = area_ratio > 0.30   # person
too_close = area_ratio > 0.08   # everything else
```

---

### 🛣️ Toby — Path Detection

Built the logic that checks whether a detected object is actually in the scooter's path, not just somewhere in the frame.

**Without SAM (default):** uses a center-of-frame heuristic — objects are considered in-path if their horizontal center falls within the middle 50% of the frame, and their bottom edge is in the lower 40% of the frame (i.e. on the ground ahead).

```python
in_path = abs(object_center_x - frame_center_x) < frame_width * 0.25
low     = y2 > frame_height * 0.6
```

**With SAM (`models/mobile_sam.pt`):** uses the [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) segmentation model to detect the actual road surface polygon. An object is only flagged if its base point falls inside that polygon — much more accurate on curved paths or angled cameras.

```python
in_path = cv2.pointPolygonTest(path_polygon, bottom_center_pt, False) >= 0
```

SAM runs every 5 frames (not every frame) to keep performance up, reusing the last known road polygon between updates.

---

### 🔔 Kenny — Alert System

Built the alert system that fires when a hazard is confirmed and manages how often alerts repeat.

- **Audio alert:** plays `alert.mp3` via pygame when a hazard is detected
- **Cooldown:** alerts can only fire every 2 seconds — prevents constant noise if an obstacle stays in frame
- **CSV logging:** every confirmed hazard is written to `hazard_log.csv` with a timestamp, object type, area ratio, and whether it was in the path

```python
if (current_time - last_alert_time) >= ALERT_COOLDOWN:
    alert_sound.play()
    last_alert_time = current_time
```

---

### 🔗 Katherine & Toby — Integration

Combined all four components into the unified web app. The final architecture runs detection in a background thread so the web server stays responsive:

```
Webcam
  │
  ▼
CameraStream thread       ← continuously captures frames in background
  │
  ▼
DetectionEngine thread    ← owns all ML inference (Ashwin + Katherine)
  ├── SAM every 5 frames       → road path polygon  (Toby)
  └── 6× YOLO every frame      → run in parallel via ThreadPoolExecutor
        │
        ▼
  Hazard evaluation             area ratio (Ashwin) + path check (Toby)
        │
        ├── Audio alert + cooldown + CSV log  (Kenny)
        └── SharedState  ←──────────────────────────┐
                                                     │
FastAPI server (main thread)                         │
  ├── GET  /              → web dashboard            │
  ├── GET  /video_feed    → MJPEG stream             │
  ├── GET  /api/status    → current hazards ─────────┘
  ├── GET  /api/log       → detection history
  ├── POST /api/toggle_model  → enable/disable models live
  └── POST /api/toggle_audio  → mute/unmute alerts
```

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

SAM makes hazard detection smarter by identifying the actual road surface rather than using a fixed center-of-frame zone.

1. Download `mobile_sam.pt` from the [MobileSAM releases page](https://github.com/ChaoningZhang/MobileSAM/releases)
2. Place it in the `models/` folder
3. Restart the app — SAM will be detected and enabled automatically

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
