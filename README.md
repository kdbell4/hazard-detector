# hazard-detector

Hazard detection using YOLO and OpenCV.

## Installation

```bash
pip install -e .
```

## Usage
run with `python -m hazard_detector webcam`

```bash
# Detect objects in an image
hazard-detector path/to/image.jpg

# Use a different model
hazard-detector path/to/image.jpg --model yolov8s.pt

# Adjust confidence threshold
hazard-detector path/to/image.jpg --conf 0.5

# Webcam detection
hazard-detector webcam

# Webcam with specific camera ID
hazard-detector webcam --camera-id 1
```