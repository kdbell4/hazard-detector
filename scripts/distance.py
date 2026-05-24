"""
Scooter hazard detector — proximity + path + position check.

Uses bounding-box area ratio, horizontal position, and vertical position
to decide if an object is a hazard. Uses Apple MPS / CUDA automatically.
Press 'q' to quit.
"""

import threading
import cv2
from ultralytics import YOLO

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

# ── Model ─────────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")
model.to(DEVICE)

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
print("Starting — press 'q' to quit.")

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    results = model(frame, imgsz=320, verbose=False)

    alert = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]

        # Proximity via bounding-box area ratio
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        too_close = area_ratio > (0.30 if label == "person" else 0.08)

        # Must be near the center of the frame (in the scooter's path)
        object_center_x = (x1 + x2) / 2
        in_path = abs(object_center_x - frame_width / 2) < frame_width * 0.25

        # Must be in the lower part of the frame (close on the ground)
        low = y2 > frame_height * 0.6

        if too_close and in_path and low:
            alert = True
            color = (0, 0, 255)
            display_label = f"HAZARD: {label}"
        else:
            color = (0, 255, 0)
            display_label = label

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if alert:
        cv2.putText(frame, "WARNING: OBSTACLE AHEAD", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Scooter Hazard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
