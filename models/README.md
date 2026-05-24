# Model Weights

This folder contains the custom-trained YOLO model weights for each hazard type.

| File | Hazard | Notes |
|------|--------|-------|
| `best.pt` | Pothole | Custom trained |
| `rocks_best.pt` | Rocks | Custom trained |
| `stairs_best.pt` | Stairs | Custom trained |
| `curb_best.pt` | Curbs | Custom trained |
| `speedbump_best.pt` | Speed bumps | Custom trained |
| `mobile_sam.pt` | Road path (SAM) | **Optional** — download separately |

## yolov8n.pt

The base YOLOv8 nano model (`yolov8n.pt`) is **not** stored here — it downloads automatically from Ultralytics on first run and is placed in the project root.

## mobile_sam.pt (optional)

SAM enables smarter path detection by segmenting the actual road surface. Without it, the app falls back to a center-of-frame heuristic which works fine.

To enable it:
1. Download `mobile_sam.pt` from the [MobileSAM releases page](https://github.com/ChaoningZhang/MobileSAM/releases)
2. Place it in this `models/` folder
