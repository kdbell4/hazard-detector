from pathlib import Path
from typing import Optional
import cv2
from ultralytics import YOLO


def get_device() -> str:
    """Return the best available inference device (mps > cuda > cpu)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, source: str, conf_threshold: float = 0.25, save: bool = True) -> list:
        results = self.model.predict(source=source, conf=conf_threshold, save=save)
        return results

    def detect_video(self, source: str, conf_threshold: float = 0.25) -> list:
        results = self.model.predict(source=source, conf=conf_threshold, save=True)
        return results

    def detect_webcam(self, camera_id: int = 0, conf_threshold: float = 0.25) -> None:
        import threading

        # Threaded camera so inference doesn't stall frame capture
        _frame = [None]
        _ret = [False]
        _lock = threading.Lock()

        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        def _capture():
            while True:
                ret, frame = cap.read()
                with _lock:
                    _ret[0], _frame[0] = ret, frame

        threading.Thread(target=_capture, daemon=True).start()
        print(f"Webcam stream started (device: {self.device}). Press 'q' to quit.")

        while True:
            with _lock:
                if _frame[0] is None:
                    continue
                frame = _frame[0].copy()

            results = self.model.predict(frame, conf=conf_threshold, imgsz=320, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO Webcam Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main(
    source: str,
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    save: bool = True,
) -> None:
    detector = YOLODetector(model_path=model)
    results = detector.detect(source=source, conf_threshold=conf, save=save)
    print(f"Processed {source}, found {len(results[0].boxes)} objects")
