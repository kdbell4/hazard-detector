"""
DetectionEngine — daemon thread that owns the webcam and all ML inference.
"""

import csv
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Prevent OpenCV's thread pool from conflicting with PyTorch's on macOS
cv2.setNumThreads(1)

from webapp.state import SharedState


# ── Device detection ──────────────────────────────────────────────────────
def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _get_device()

PROJECT_ROOT = Path(__file__).parent.parent

HAZARD_IDS = {0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 36}

MODEL_REGISTRY = {
    "base":      {"file": "yolov8n.pt",              "color": (0, 255, 0),   "conf": 0.50, "use_hazard_filter": True},
    "pothole":   {"file": "models/best.pt",           "color": (0, 0, 255),   "conf": 0.80, "use_hazard_filter": False},
    "rocks":     {"file": "models/rocks_best.pt",     "color": (0, 165, 255), "conf": 0.80, "use_hazard_filter": False},
    "stairs":    {"file": "models/stairs_best.pt",    "color": (255, 0, 0),   "conf": 0.80, "use_hazard_filter": False},
    "curb":      {"file": "models/curb_best.pt",      "color": (255, 255, 0), "conf": 0.80, "use_hazard_filter": False},
    "speedbump": {"file": "models/speedbump_best.pt", "color": (255, 0, 255), "conf": 0.80, "use_hazard_filter": False},
}

PROXIMITY_PERSON  = 0.30
PROXIMITY_DEFAULT = 0.08
LABEL_COOLDOWN    = 8.0
SAM_INTERVAL      = 5

# Custom models run every Nth frame; base runs every frame.
# Potholes/rocks/curbs etc. are mostly stationary so cached boxes stay accurate.
CUSTOM_MODEL_INTERVAL = 3

# Resize frames to this width before inference to reduce GPU data transfer.
INFERENCE_WIDTH = 640


class DetectionEngine(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self._state      = state
        self._stop_event = threading.Event()

        self._models: dict = {}
        self._sam          = None
        self._sam_available = False

        self._path_polygon  = None
        self._frame_count   = 0
        self._label_alert_times: dict = {}
        self._cached_custom_detections: list = []
        self._fps_times: deque = deque(maxlen=30)

        # Persistent thread pool — created once, reused every frame
        self._executor = ThreadPoolExecutor(max_workers=6)

        self._alert_sound = None
        self._log_path    = PROJECT_ROOT / "hazard_log.csv"

    def stop(self):
        self._stop_event.set()
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # Init                                                                #
    # ------------------------------------------------------------------ #

    def _load_models(self):
        from ultralytics import YOLO, SAM

        print(f"Loading YOLO models… (device: {DEVICE})")
        for name, cfg in MODEL_REGISTRY.items():
            model_path = PROJECT_ROOT / cfg["file"]
            if not model_path.exists():
                print(f"  [WARN] {model_path.name} not found — skipping '{name}'")
                continue
            try:
                m = YOLO(str(model_path))
                m.to(DEVICE)
                self._models[name] = m
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

        sam_path = PROJECT_ROOT / "models" / "mobile_sam.pt"
        if sam_path.exists():
            try:
                self._sam = SAM(str(sam_path))
                self._sam_available = True
                print("  ✓ SAM")
            except Exception as e:
                print(f"  [WARN] SAM failed: {e}")
        else:
            print("  [WARN] models/mobile_sam.pt not found — path detection disabled")

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            alert_path = PROJECT_ROOT / "alert.mp3"
            if alert_path.exists():
                self._alert_sound = pygame.mixer.Sound(str(alert_path))
                print("  ✓ Audio")
            else:
                print("  [WARN] alert.mp3 not found")
        except Exception as e:
            print(f"  [WARN] Audio init failed: {e}")

    def _init_csv(self):
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "object_type", "area_ratio", "in_path", "model"])
        print(f"  ✓ CSV log → {self._log_path}")

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        print("\nInitialising DetectionEngine…")
        self._load_models()
        self._init_audio()
        self._init_csv()

        source_str = os.environ.get("HAZARD_SOURCE", "0")
        try:
            source = int(source_str)
        except ValueError:
            source = source_str

        print(f"Opening {'camera ' + str(source) if isinstance(source, int) else source}…\n")

        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Could not open '{source}'")
            return

        is_video_file = isinstance(source, str)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                else:
                    time.sleep(0.05)
                continue

            # Resize for inference — reduces GPU data transfer cost.
            # All inference and drawing happens on the smaller frame.
            h_orig, w_orig = frame.shape[:2]
            if w_orig > INFERENCE_WIDTH:
                scale = INFERENCE_WIDTH / w_orig
                frame = cv2.resize(frame, (INFERENCE_WIDTH, int(h_orig * scale)))

            h, w       = frame.shape[:2]
            annotated  = frame.copy()
            current_time = time.time()

            # SAM path segmentation
            if self._sam_available and self._frame_count % SAM_INTERVAL == 0:
                self._update_path_polygon(frame, h, w)
            if self._path_polygon is not None:
                pts = self._path_polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 100), thickness=2)

            # Inference: base every frame, custom models every 3rd frame
            run_custom   = (self._frame_count % CUSTOM_MODEL_INTERVAL == 0)
            all_detections = self._run_models(frame, run_custom)

            # Hazard evaluation
            hazards_this_frame: list[dict] = []
            log_entries: list[dict] = []

            for det in all_detections:
                x1, y1, x2, y2 = det["xyxy"]
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                label      = det["label"]

                too_close     = area_ratio > (PROXIMITY_PERSON if label == "person" else PROXIMITY_DEFAULT)
                bottom_center = (int((x1 + x2) / 2), y2)
                in_path       = (
                    cv2.pointPolygonTest(self._path_polygon, bottom_center, False) >= 0
                    if self._path_polygon is not None else True
                )
                low       = y2 > h * 0.6
                is_hazard = too_close and in_path and low

                color       = (0, 0, 255) if is_hazard else det["color"]
                display_lbl = f"{'HAZARD: ' if is_hazard else ''}{label} {det['conf']:.0%}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, display_lbl, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if is_hazard:
                    hazards_this_frame.append({
                        "label": label, "model": det["model_name"],
                        "area_pct": round(area_ratio * 100, 1), "in_path": in_path,
                    })
                    log_entries.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "object_type": label, "area_ratio": f"{area_ratio:.4f}",
                        "in_path": str(in_path), "model": det["model_name"],
                    })

            # Alerts
            if hazards_this_frame:
                for hazard in hazards_this_frame:
                    lbl = hazard["label"]
                    if current_time - self._label_alert_times.get(lbl, 0) >= LABEL_COOLDOWN:
                        self._play_audio()
                        self._label_alert_times[lbl] = current_time
                        break
                self._write_csv(log_entries)
                for entry in log_entries:
                    self._state.add_log_entry(entry)
                cv2.putText(annotated, "⚠ HAZARD DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # Update shared state
            self._state.set_current_hazards(hazards_this_frame)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
            self._state.set_frame(jpeg.tobytes())

            # FPS tracking
            self._fps_times.append(current_time)
            if len(self._fps_times) >= 2:
                elapsed = self._fps_times[-1] - self._fps_times[0]
                if elapsed > 0:
                    self._state.set_fps(len(self._fps_times) / elapsed)

            self._frame_count += 1

        cap.release()
        print("DetectionEngine stopped.")

    # ------------------------------------------------------------------ #
    # SAM                                                                 #
    # ------------------------------------------------------------------ #

    def _update_path_polygon(self, frame, h, w):
        try:
            points  = [[2 * w // 5, h], [w // 2, h], [3 * w // 5, h]]
            results = self._sam.predict(frame, conf=0.25, points=points, verbose=False)
            if results[0].masks and results[0].masks.xy:
                self._path_polygon = max(results[0].masks.xy, key=cv2.contourArea)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def _run_one_model(self, name: str, cfg: dict, frame) -> list[dict]:
        detections = []
        try:
            results = self._models[name](frame, conf=cfg["conf"], imgsz=320, verbose=False)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cfg["use_hazard_filter"] and cls_id not in HAZARD_IDS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "model_name": name,
                    "label":      results[0].names[cls_id],
                    "conf":       float(box.conf[0]),
                    "xyxy":       (x1, y1, x2, y2),
                    "color":      cfg["color"],
                })
        except Exception as e:
            print(f"[WARN] {name}: {e}")
        return detections

    def _run_models(self, frame, run_custom: bool) -> list[dict]:
        enabled = {
            name: cfg for name, cfg in MODEL_REGISTRY.items()
            if name in self._models and self._state.is_model_enabled(name)
        }
        to_run = {
            name: cfg for name, cfg in enabled.items()
            if name == "base" or run_custom
        }
        if not to_run:
            return []

        base_results, custom_results = [], []

        if DEVICE == "mps":
            # MPS is not thread-safe — concurrent kernel launches from
            # multiple threads cause a segfault. Run models serially instead.
            for name, cfg in to_run.items():
                dets = self._run_one_model(name, cfg, frame)
                (base_results if name == "base" else custom_results).extend(dets)
        else:
            # CPU / CUDA: safe to parallelise
            futures = {
                self._executor.submit(self._run_one_model, name, cfg, frame): name
                for name, cfg in to_run.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                dets = future.result()
                (base_results if name == "base" else custom_results).extend(dets)

        if run_custom:
            self._cached_custom_detections = custom_results

        return base_results + self._cached_custom_detections

    # ------------------------------------------------------------------ #
    # Audio / CSV                                                         #
    # ------------------------------------------------------------------ #

    def _play_audio(self):
        if not self._state.is_audio_enabled() or self._alert_sound is None:
            return
        try:
            import pygame
            if not pygame.mixer.get_busy():
                self._alert_sound.play()
        except Exception:
            pass

    def _write_csv(self, entries):
        try:
            with open(self._log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "object_type", "area_ratio", "in_path", "model"])
                for entry in entries:
                    writer.writerow(entry)
        except Exception as e:
            print(f"[WARN] CSV write failed: {e}")
