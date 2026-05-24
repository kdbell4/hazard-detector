"""
SharedState — thread-safe data store between the DetectionEngine daemon thread
and the FastAPI async event loop.
"""

import threading
import time
from collections import deque
from typing import Any


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()

        # Video
        self.latest_frame: bytes = b""

        # Active hazards in the most recent frame
        self.current_hazards: list[dict] = []

        # Rolling log of last 100 detection events
        self.recent_log: deque = deque(maxlen=100)

        # Per-model on/off flags — rocks starts disabled (matches models.py)
        self.model_enabled: dict[str, bool] = {
            "base":      True,
            "pothole":   True,
            "rocks":     False,
            "stairs":    True,
            "curb":      True,
            "speedbump": True,
        }

        self.audio_enabled: bool = True

        # Stats
        self.total_detections: int = 0
        self.start_time: float = time.time()
        self.fps: float = 0.0

    # ------------------------------------------------------------------ #
    # Frame                                                               #
    # ------------------------------------------------------------------ #

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self.latest_frame = jpeg_bytes

    def get_frame(self) -> bytes:
        with self._lock:
            return self.latest_frame

    # ------------------------------------------------------------------ #
    # Hazards                                                              #
    # ------------------------------------------------------------------ #

    def set_current_hazards(self, hazards: list[dict]) -> None:
        with self._lock:
            self.current_hazards = hazards

    # ------------------------------------------------------------------ #
    # Log                                                                  #
    # ------------------------------------------------------------------ #

    def add_log_entry(self, entry: dict) -> None:
        with self._lock:
            self.recent_log.appendleft(entry)
            self.total_detections += 1

    def get_log_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.recent_log)

    # ------------------------------------------------------------------ #
    # Model / audio toggles                                               #
    # ------------------------------------------------------------------ #

    def toggle_model(self, model_name: str, enabled: bool) -> bool:
        """Returns True if model_name was valid, False otherwise."""
        with self._lock:
            if model_name not in self.model_enabled:
                return False
            self.model_enabled[model_name] = enabled
            return True

    def toggle_audio(self, enabled: bool) -> None:
        with self._lock:
            self.audio_enabled = enabled

    def is_model_enabled(self, model_name: str) -> bool:
        with self._lock:
            return self.model_enabled.get(model_name, False)

    def is_audio_enabled(self) -> bool:
        with self._lock:
            return self.audio_enabled

    # ------------------------------------------------------------------ #
    # FPS                                                                  #
    # ------------------------------------------------------------------ #

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self.fps = fps

    # ------------------------------------------------------------------ #
    # Snapshot for API responses                                          #
    # ------------------------------------------------------------------ #

    def get_status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hazards":      list(self.current_hazards),
                "model_states": dict(self.model_enabled),
                "audio_enabled": self.audio_enabled,
                "fps":          round(self.fps, 1),
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "total_detections": self.total_detections,
            }
