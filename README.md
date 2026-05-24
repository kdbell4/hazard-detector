# Hazard Detector

Real-time hazard detection for electric scooters. A webcam feed is analyzed using computer vision to detect obstacles in the rider's path and trigger an audio alert before they become a danger.

---

## What it does

Electric scooters are low to the ground and move fast, which makes obstacles like potholes, rocks, and curbs genuinely dangerous — especially when riders are focused on traffic and not the road surface. This project is a real-time hazard detection system designed to act as a second set of eyes.

A webcam is mounted on the scooter and streams live video into the app. The app constantly analyzes the video to identify anything that could be a hazard — potholes, rocks, stairs, curbs, speed bumps, people, animals, bikes, and cars. When something dangerous is spotted, it plays an audio alert so the rider knows to slow down or steer around it, without having to look at a screen.

The tricky part is avoiding false alarms. Just because a car appears in the frame doesn't mean it's about to be hit — it could be a parked car off to the side, or a vehicle far down the road. So before triggering an alert, the app checks three things: is the object large enough in the frame to be close by, is it directly ahead in the scooter's path, and is it low enough in the frame to actually be on the ground nearby? All three have to be true before an alarm fires.

The app also runs a road segmentation model that figures out exactly where the road surface is in front of the scooter. This makes the path check smarter — instead of just looking at whether something is in the center of the frame, it checks whether the object is sitting on the actual road ahead.

Everything is accessible through a web dashboard that shows the live camera feed with detected objects highlighted, which models are active, a log of past detections, and controls to mute the audio or toggle individual models on and off.

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
