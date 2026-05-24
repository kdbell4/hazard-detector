# hazard-detector

Real-time hazard detection for scooters using YOLOv8 and OpenCV.

Detects people, potholes, rocks, stairs, curbs, and speedbumps from a webcam feed and plays an audio alert when a hazard enters the rider's path.

## Requirements

- Python 3.8+
- A webcam (built-in or USB)
- **Recommended:** Mac with Apple Silicon (M1/M2/M3) — uses the Metal GPU automatically for 3–5× faster inference

---

## Install

```bash
git clone <repo-url>
cd hazard-detector
pip install -e .
```

Or if you just want to run a single script without installing:

```bash
pip install ultralytics opencv-python pygame
```

---

## How to run

### Option 1 — Web dashboard (easiest, no CV2 window required)

```bash
python run_webapp.py
```

Then open **http://localhost:8000** in any browser. You'll see a live video feed, hazard alerts, detection log, and toggles for each model. Works great for sharing with others — anyone on your local network can view it at `http://<your-ip>:8000`.

### Option 2 — Webcam window (terminal)

```bash
python -m hazard_detector webcam
```

Opens an OpenCV window with live detection. Press `q` to quit.

If your webcam isn't found, try camera ID 1:

```bash
python -m hazard_detector webcam --camera-id 1
```

### Option 3 — Video file (no webcam needed — great for testing)

```bash
python -m hazard_detector path/to/video.mp4
```

This works with any `.mp4`, `.mov`, or other video file. No webcam required — ideal for trying the app without any hardware setup.

**Free test videos:** Search YouTube for dashcam or cycling footage and download with `yt-dlp`:
```bash
pip install yt-dlp
yt-dlp -f mp4 "https://www.youtube.com/watch?v=<video-id>" -o test_video.mp4
python -m hazard_detector test_video.mp4
```

---

## Easy ways to try it without setting everything up

| Method | Effort | Notes |
|--------|--------|-------|
| **Web dashboard** | Low | Just `pip install -e .` + `python run_webapp.py`, view in browser |
| **Video file mode** | Low | No webcam needed — pass any `.mp4` |
| **Webcam mode** | Medium | Requires a working webcam and OpenCV display |
| **Share web dashboard** | Low | Others on your Wi-Fi can open `http://<your-ip>:8000` |

---

## Model files

| Model | File | What it detects | Auto-downloads? |
|-------|------|-----------------|-----------------|
| Base (YOLOv8n) | `yolov8n.pt` | People, cars, bikes, animals | ✅ Yes |
| Pothole | `best.pt` | Potholes | Included in repo |
| Rocks | `rocks_best.pt` | Rocks | Included in repo |
| Stairs | `stairs_best.pt` | Stairs | Included in repo |
| Curb | `curb_best.pt` | Curbs | Included in repo |
| Speed bump | `speedbump_best.pt` | Speed bumps | Included in repo |
| SAM (optional) | `mobile_sam.pt` | Road path segmentation | Manual download |

### Optional: SAM path detection

SAM makes the hazard zone smarter by segmenting the actual road surface instead of using a fixed center-of-frame box. Without it, the app falls back to a center heuristic that works fine.

Download `mobile_sam.pt` from [MobileSAM releases](https://github.com/ChaoningZhang/MobileSAM/releases) and place it in this directory.

---

## Performance

| Hardware | Expected FPS |
|----------|-------------|
| Mac Apple Silicon (M1/M2/M3) | 15–25 FPS |
| Mac Intel / Windows CPU | 4–10 FPS |
| NVIDIA GPU | 20–40 FPS |

The app auto-detects your hardware and uses MPS (Apple) or CUDA (NVIDIA) automatically. To improve speed:
- Disable unused models in the web dashboard
- Use the base model only (`python -m hazard_detector webcam --model yolov8n.pt`)

---

## Troubleshooting

**Camera not opening:**
```bash
# Try camera index 0 or 1
python -m hazard_detector webcam --camera-id 0
python -m hazard_detector webcam --camera-id 1
```

**`pygame` audio error on first run:**
```bash
pip install pygame
```

**Missing model weights:**
All `.pt` files except `yolov8n.pt` (auto-downloaded) and `mobile_sam.pt` (optional) must be in the `hazard-detector/` root directory. The app prints a warning and skips any missing model rather than crashing.
