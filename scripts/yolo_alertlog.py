"""
Hazard detector with audio alerts and CSV logging.

Combines YOLO detection + SAM path segmentation.
Uses Apple MPS / CUDA automatically when available.
Press 'q' to quit.
"""

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, SAM
import pygame

# ── Device detection ──────────────────────────────────────────────────────
def get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ── Audio setup ───────────────────────────────────────────────────────────
pygame.mixer.init()
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
alert_path = PROJECT_ROOT / "alert.mp3"
alert_sound = pygame.mixer.Sound(str(alert_path)) if alert_path.exists() else None
if not alert_sound:
    print("[WARN] alert.mp3 not found — audio alerts disabled")

# ── CSV logging ───────────────────────────────────────────────────────────
LOG_FILE_PATH = PROJECT_ROOT / "hazard_log.csv"
print(f"Saving log to: {LOG_FILE_PATH}")
if not LOG_FILE_PATH.exists():
    with open(LOG_FILE_PATH, mode="w", newline="") as file:
        csv.writer(file).writerow(["timestamp", "object_type", "area_ratio", "in_path"])

# ── Models ────────────────────────────────────────────────────────────────
print("Loading models…")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(DEVICE)

sam_path = PROJECT_ROOT / "models" / "mobile_sam.pt"

sam_model = None
if sam_path.exists():
    sam_model = SAM(str(sam_path))
    print("  ✓ SAM loaded")
else:
    print("  [WARN] mobile_sam.pt not found — path detection disabled (using center heuristic)")

ALERT_COOLDOWN_SECONDS = 2.0
last_alert_time = 0.0
frame_count = 0
SAM_INTERVAL = 5
path_polygon = None

# ── Threaded camera ───────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while True:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        self.cap.release()


# ── Main loop ─────────────────────────────────────────────────────────────
stream = CameraStream(camera_id=0)
print("Initialization complete! Starting webcam feed…")

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    current_time = time.time()
    alert_triggered_this_frame = False
    hazards_to_log = []
    annotated_frame = frame.copy()

    # ── SAM path segmentation (every N frames) ────────────────────────────
    if sam_model is not None and frame_count % SAM_INTERVAL == 0:
        try:
            ctrdots = [[x, h] for x in [2 * w // 5, w // 2, 3 * w // 5]]
            sam_results = sam_model.predict(frame, conf=0.25, points=ctrdots, verbose=False)
            if sam_results[0].masks and sam_results[0].masks.xy:
                polygons = sam_results[0].masks.xy
                path_polygon = max(polygons, key=cv2.contourArea)
        except Exception as e:
            pass  # non-fatal — keep last polygon

    if path_polygon is not None:
        pts = path_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 100), thickness=2)

    # ── YOLO detection ────────────────────────────────────────────────────
    yolo_results = yolo_model(frame, imgsz=320, verbose=False)

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = yolo_model.names[int(box.cls)]

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / (w * h)

        too_close = area_ratio > (0.30 if label == "person" else 0.08)

        # Path intersection
        bottom_center_pt = (int((x1 + x2) / 2), y2)
        if path_polygon is not None:
            in_path = cv2.pointPolygonTest(path_polygon, bottom_center_pt, False) >= 0
        else:
            # Fallback: center-of-frame heuristic
            in_path = abs(bottom_center_pt[0] - w / 2) < w * 0.25

        low = y2 > (h * 0.6)

        if too_close and in_path and low:
            alert_triggered_this_frame = True
            color = (0, 0, 255)
            display_label = f"HAZARD: {label}"
            hazards_to_log.append((label, area_ratio, in_path))
        else:
            color = (255, 0, 0)
            display_label = label

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Alerts & logging ──────────────────────────────────────────────────
    if alert_triggered_this_frame:
        if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
            if alert_sound and not pygame.mixer.get_busy():
                alert_sound.play()
            last_alert_time = current_time

        cv2.putText(annotated_frame, "WARNING: OBSTACLE AHEAD", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            for obj_type, ratio, path_status in hazards_to_log:
                writer.writerow([timestamp_str, obj_type, f"{ratio:.4f}", path_status])

    cv2.imshow("Advanced Scooter Hazard Detection", annotated_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
