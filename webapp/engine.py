"""
DetectionEngine — daemon thread that owns the webcam and all ML inference.

Combines logic from:
  - models.py        (multi-model YOLO, color map, HAZARD_IDS, conf levels)
  - yolo_alertlog.py (SAM path segmentation, proximity checks, audio, CSV logging)
  - distance.py      (area-ratio proximity thresholds)
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

# ─────────────────────────────────────────────
# Project root (two levels up from this file)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────
# Model registry: name → {file, color, conf, filter}
# ─────────────────────────────────────────────
HAZARD_IDS = {0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 36}

MODEL_REGISTRY = {
    "base":      {"file": "yolov8n.pt",              "color": (0, 255, 0),   "conf": 0.50, "use_hazard_filter": True},
    "pothole":   {"file": "models/best.pt",           "color": (0, 0, 255),   "conf": 0.80, "use_hazard_filter": False},
    "rocks":     {"file": "models/rocks_best.pt",     "color": (0, 165, 255), "conf": 0.80, "use_hazard_filter": False},
    "stairs":    {"file": "models/stairs_best.pt",    "color": (255, 0, 0),   "conf": 0.80, "use_hazard_filter": False},
    "curb":      {"file": "models/curb_best.pt",      "color": (255, 255, 0), "conf": 0.80, "use_hazard_filter": False},
    "speedbump": {"file": "models/speedbump_best.pt", "color": (255, 0, 255), "conf": 0.80, "use_hazard_filter": False},
}

# Proximity thresholds (area_ratio)
PROXIMITY_PERSON  = 0.30
PROXIMITY_DEFAULT = 0.08

# Per-label alert cooldown — how long before the same object type can alert again
LABEL_COOLDOWN = 8.0

# SAM segmentation interval (every N frames)
SAM_INTERVAL = 5


class DetectionEngine(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self._state = state
        self._stop_event = threading.Event()

        # Loaded model objects {name: YOLO instance}
        self._models: dict = {}
        self._sam = None
        self._sam_available = False

        # Detection state
        self._path_polygon = None
        self._frame_count = 0
        self._label_alert_times: dict = {}   # label -> last time it was alerted
        self._fps_times: deque = deque(maxlen=30)

        # Audio
        self._alert_sound = None

        # CSV
        self._log_path = PROJECT_ROOT / "hazard_log.csv"

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Initialisation helpers (called inside run() so they're on our thread)
    # ------------------------------------------------------------------ #

    def _load_models(self):
        from ultralytics import YOLO, SAM  # deferred import for thread safety

        print(f"Loading YOLO models… (device: {DEVICE})")
        for name, cfg in MODEL_REGISTRY.items():
            model_path = PROJECT_ROOT / cfg["file"]
            if not model_path.exists():
                print(f"  [WARN] Model file not found: {model_path} — skipping '{name}'")
                continue
            try:
                m = YOLO(str(model_path))
                m.to(DEVICE)
                self._models[name] = m
                print(f"  ✓ {name} ({cfg['file']})")
            except Exception as e:
                print(f"  [ERROR] Could not load '{name}': {e}")

        sam_path = PROJECT_ROOT / "models" / "mobile_sam.pt"
        if sam_path.exists():
            try:
                self._sam = SAM(str(sam_path))
                self._sam_available = True
                print("  ✓ SAM (models/mobile_sam.pt)")
            except Exception as e:
                print(f"  [WARN] SAM load failed: {e} — path detection disabled")
        else:
            print(
                "  [WARN] models/mobile_sam.pt not found — SAM path detection disabled.\n"
                "         Download it from: https://github.com/ChaoningZhang/MobileSAM\n"
                "         Place mobile_sam.pt in the models/ folder to enable path awareness."
            )

    def _init_audio(self):
        try:
            # Prevent SDL display init — cv2 and pygame both bundle SDL2 and
            # conflict on macOS, causing a segfault when both try to own the
            # display. Setting these before any pygame import stops that.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "coreaudio")
            import pygame
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            alert_path = PROJECT_ROOT / "alert.mp3"
            if alert_path.exists():
                self._alert_sound = pygame.mixer.Sound(str(alert_path))
                print("  ✓ Audio (alert.mp3)")
            else:
                print("  [WARN] alert.mp3 not found — audio alerts disabled")
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

        # Resolve source: env var → int (camera ID) or str (video file path)
        source_str = os.environ.get("HAZARD_SOURCE", "0")
        try:
            source = int(source_str)
            source_label = f"camera {source}"
        except ValueError:
            source = source_str
            source_label = f"video file: {source}"

        print(f"Initialisation complete. Opening {source_label}…\n")

        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Could not open source '{source}' — engine shutting down.")
            return

        is_video_file = isinstance(source, str)

        # For video files: track real-time position so we can skip frames
        # and keep playback speed in sync with the original video FPS.
        video_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 0
        video_start_wall = time.time()
        video_frames_read = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    # Loop back to the start and reset timing
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_start_wall = time.time()
                    video_frames_read = 0
                    continue
                time.sleep(0.05)
                continue

            video_frames_read += 1

            # ── Frame skipping for video files ────────────────────────── #
            # If inference is slower than the video's native FPS, skip ahead
            # so playback stays in sync with real time.
            if is_video_file and video_fps > 0:
                expected = int((time.time() - video_start_wall) * video_fps)
                skip = expected - video_frames_read
                for _ in range(max(0, min(skip, 8))):   # cap at 8 skipped frames
                    cap.read()
                    video_frames_read += 1

            h, w = frame.shape[:2]
            annotated = frame.copy()
            current_time = time.time()

            # ── SAM path segmentation ──────────────────────────────── #
            if self._sam_available and self._frame_count % SAM_INTERVAL == 0:
                self._update_path_polygon(frame, h, w)

            if self._path_polygon is not None:
                pts = self._path_polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 100), thickness=2)

            # ── YOLO inference across enabled models ───────────────── #
            all_detections = self._run_all_models(frame)

            # ── Per-detection hazard evaluation ───────────────────── #
            hazards_this_frame: list[dict] = []
            log_entries: list[dict] = []

            for det in all_detections:
                x1, y1, x2, y2 = det["xyxy"]
                box_area  = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / (w * h)
                label      = det["label"]

                too_close = area_ratio > (PROXIMITY_PERSON if label == "person" else PROXIMITY_DEFAULT)

                # Path check
                bottom_center = (int((x1 + x2) / 2), y2)
                if self._path_polygon is not None:
                    result   = cv2.pointPolygonTest(self._path_polygon, bottom_center, False)
                    in_path  = result >= 0
                else:
                    in_path  = True  # conservative when SAM unavailable

                low = y2 > h * 0.6

                is_hazard = too_close and in_path and low

                color       = (0, 0, 255) if is_hazard else det["color"]
                prefix      = "HAZARD: "  if is_hazard else ""
                display_lbl = f"{prefix}{label} {det['conf']:.0%}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, display_lbl,
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if is_hazard:
                    hazard_info = {
                        "label":      label,
                        "model":      det["model_name"],
                        "area_pct":   round(area_ratio * 100, 1),
                        "in_path":    in_path,
                    }
                    hazards_this_frame.append(hazard_info)
                    log_entries.append({
                        "timestamp":   datetime.now().strftime("%H:%M:%S"),
                        "object_type": label,
                        "area_ratio":  f"{area_ratio:.4f}",
                        "in_path":     str(in_path),
                        "model":       det["model_name"],
                    })

            # ── Alerts & logging ───────────────────────────────────── #
            if hazards_this_frame:
                # Fire audio only for labels that haven't alerted recently.
                # This means each object type alerts once when it first
                # appears, then goes quiet until it's been gone long enough.
                for hazard in hazards_this_frame:
                    lbl = hazard["label"]
                    last = self._label_alert_times.get(lbl, 0)
                    if current_time - last >= LABEL_COOLDOWN:
                        self._play_audio()
                        self._label_alert_times[lbl] = current_time
                        break   # one sound at a time; next new label fires next frame

                self._write_csv(log_entries)
                for entry in log_entries:
                    self._state.add_log_entry(entry)

                cv2.putText(annotated, "⚠ HAZARD DETECTED",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # ── Update shared state ────────────────────────────────── #
            self._state.set_current_hazards(hazards_this_frame)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._state.set_frame(jpeg.tobytes())

            # ── FPS tracking ──────────────────────────────────────── #
            self._fps_times.append(current_time)
            if len(self._fps_times) >= 2:
                elapsed = self._fps_times[-1] - self._fps_times[0]
                if elapsed > 0:
                    self._state.set_fps(len(self._fps_times) / elapsed)

            self._frame_count += 1

        cap.release()
        print("DetectionEngine stopped.")

    # ------------------------------------------------------------------ #
    # SAM helper                                                          #
    # ------------------------------------------------------------------ #

    def _update_path_polygon(self, frame, h: int, w: int):
        try:
            points = [[2 * w // 5, h], [w // 2, h], [3 * w // 5, h]]
            results = self._sam.predict(frame, conf=0.25, points=points, verbose=False)
            if results[0].masks and results[0].masks.xy:
                polygons = results[0].masks.xy
                self._path_polygon = max(polygons, key=cv2.contourArea)
        except Exception as e:
            pass  # SAM failures are non-fatal; keep last polygon

    # ------------------------------------------------------------------ #
    # Multi-model YOLO runner                                             #
    # ------------------------------------------------------------------ #

    def _run_one_model(self, name: str, cfg: dict, frame) -> list[dict]:
        """Run a single model and return its detections. Thread-safe."""
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
            print(f"[WARN] Model '{name}' inference error: {e}")
        return detections

    def _run_all_models(self, frame) -> list[dict]:
        """Run all enabled models in parallel and merge detections."""
        enabled = [
            (name, cfg) for name, cfg in MODEL_REGISTRY.items()
            if name in self._models and self._state.is_model_enabled(name)
        ]
        if not enabled:
            return []

        detections = []
        with ThreadPoolExecutor(max_workers=len(enabled)) as executor:
            futures = {
                executor.submit(self._run_one_model, name, cfg, frame): name
                for name, cfg in enabled
            }
            for future in as_completed(futures):
                detections.extend(future.result())
        return detections

    # ------------------------------------------------------------------ #
    # Audio helper                                                        #
    # ------------------------------------------------------------------ #

    def _play_audio(self):
        if not self._state.is_audio_enabled():
            return
        if self._alert_sound is None:
            return
        try:
            import pygame
            if not pygame.mixer.get_busy():
                self._alert_sound.play()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # CSV helper                                                          #
    # ------------------------------------------------------------------ #

    def _write_csv(self, entries: list[dict]):
        try:
            with open(self._log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "object_type", "area_ratio", "in_path", "model"])
                for entry in entries:
                    writer.writerow(entry)
        except Exception as e:
            print(f"[WARN] CSV write failed: {e}")
