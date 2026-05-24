"""
Multi-model YOLO hazard detector (standalone script).

Runs 5 YOLO models in parallel on each frame and highlights hazards.
Uses Apple MPS / CUDA automatically when available.
Press 'q' to quit.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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

# ── Model paths (project root is one level up from scripts/) ─────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

HAZARD_IDS = {0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 36}

MODEL_CONFIGS = [
    {"file": PROJECT_ROOT / "yolov8m.pt",        "color": (0, 255, 0),   "conf": 0.80, "hazard_filter": True},
    {"file": MODELS_DIR   / "best.pt",            "color": (0, 0, 255),   "conf": 0.80, "hazard_filter": False},
    # {"file": MODELS_DIR / "rocks_best.pt",      "color": (0, 165, 255), "conf": 0.80, "hazard_filter": False},  # disabled by default
    {"file": MODELS_DIR   / "stairs_best.pt",     "color": (255, 0, 0),   "conf": 0.80, "hazard_filter": False},
    {"file": MODELS_DIR   / "curb_best.pt",       "color": (255, 255, 0), "conf": 0.80, "hazard_filter": False},
    {"file": MODELS_DIR   / "speedbump_best.pt",  "color": (255, 0, 255), "conf": 0.80, "hazard_filter": False},
]

# Load only models whose files exist
models = []
for cfg in MODEL_CONFIGS:
    if cfg["file"].exists():
        m = YOLO(str(cfg["file"]))
        m.to(DEVICE)
        models.append({"model": m, **cfg})
        print(f"  ✓ Loaded {cfg['file'].name}")
    else:
        print(f"  [WARN] Not found, skipping: {cfg['file'].name}")

# ── Threaded camera (keeps buffer fresh while inference runs) ─────────────
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


# ── Parallel inference helper ─────────────────────────────────────────────
def run_model(cfg: dict, frame, frame_area: float) -> list[dict]:
    """Run one model and return a list of detection dicts."""
    detections = []
    try:
        results = cfg["model"](frame, conf=cfg["conf"], imgsz=320, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cfg["hazard_filter"] and cls_id not in HAZARD_IDS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": results[0].names[cls_id],
                "xyxy":  (x1, y1, x2, y2),
                "color": cfg["color"],
            })
    except Exception as e:
        print(f"[WARN] Inference error: {e}")
    return detections


# ── Main loop ─────────────────────────────────────────────────────────────
stream = CameraStream(camera_id=0)

print("Starting — press 'q' to quit.")

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    frame_area = h * w
    annotated = frame.copy()

    # Run all models in parallel
    all_detections = []
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(run_model, cfg, frame, frame_area): cfg for cfg in models}
        for future in as_completed(futures):
            all_detections.extend(future.result())

    # Draw detections
    for det in all_detections:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), det["color"], 2)
        cv2.putText(annotated, det["label"], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, det["color"], 2)

    cv2.imshow("Hazard Detection", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
