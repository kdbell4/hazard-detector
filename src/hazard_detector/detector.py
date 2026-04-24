from pathlib import Path
from typing import Optional
import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, source: str, conf_threshold: float = 0.25, save: bool = True) -> list:
        results = self.model.predict(source=source, conf=conf_threshold, save=save)
        return results

    def detect_video(self, source: str, conf_threshold: float = 0.25) -> list:
        results = self.model.predict(source=source, conf=conf_threshold, save=True)
        return results

    def detect_webcam(self, camera_id: int = 0, conf_threshold: float = 0.25) -> None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print(f"Webcam stream started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            results = self.model.predict(frame, conf=conf_threshold, verbose=False)

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