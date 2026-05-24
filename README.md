# Hazard Detector

Real-time hazard detection for electric scooters. A webcam feed is analyzed using computer vision to detect obstacles in the rider's path and trigger an audio alert before they become a danger.

---

## How it works

Each frame from the webcam is run through up to six YOLO models in parallel. Every detected object is checked against three conditions — is it close enough, is it directly ahead, and is it on the ground in front of the scooter? Only if all three are true does it trigger an alert. Detections are logged to a CSV file and streamed live to a web dashboard.

---

## Team

**Toby** — Built the core hazard detection logic (deciding when an object is actually a threat based on size, position, and proximity) and combined everyone's work into the final web app.

**Ashwin** — Identified what YOLOv8 can detect out of the box and trained custom models for hazards it couldn't handle — potholes, rocks, stairs, curbs, and speed bumps — sourcing public datasets from Roboflow and collecting custom data.

**Kenny** — Built the path detection system using MobileSAM to segment the road surface in real time, so the app only alerts on objects that are actually in the scooter's path rather than off to the side.

**Katherine** — Built the alert system: audio playback when a hazard is detected, a 2-second cooldown to avoid constant beeping, and CSV logging of every detection with timestamps.

---

## Install

```bash
git clone https://github.com/kdbell4/hazard-detector
cd hazard-detector
pip install -e .
```

---

## Run

**Web dashboard** (recommended — open in any browser):
```bash
python run_webapp.py
# then open http://localhost:8000
```

**Webcam window:**
```bash
python -m hazard_detector webcam
# press q to quit — try --camera-id 1 if camera isn't found
```

**Video file** (no webcam needed):
```bash
python -m hazard_detector path/to/video.mp4
```

---

## Models

| Hazard | Detected by |
|--------|-------------|
| People, bikes, cars, animals | YOLOv8 base (auto-downloads) |
| Potholes | `models/best.pt` |
| Rocks | `models/rocks_best.pt` |
| Stairs | `models/stairs_best.pt` |
| Curbs | `models/curb_best.pt` |
| Speed bumps | `models/speedbump_best.pt` |

An optional SAM model (`models/mobile_sam.pt`) enables smarter road segmentation. Download from [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/releases) and place in `models/` — the app detects it automatically.
