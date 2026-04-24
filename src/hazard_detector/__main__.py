import argparse
import sys
from pathlib import Path

from .detector import YOLODetector, main as detector_main


def cli():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Demo")
    parser.add_argument("source", type=str, nargs="?", help="Path to image, video, or 'webcam' for camera")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Disable saving results")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam camera ID")
    args = parser.parse_args()

    if args.source == "webcam" or args.source is None:
        detector = YOLODetector(model_path=args.model)
        detector.detect_webcam(camera_id=args.camera_id, conf_threshold=args.conf)
    else:
        detector_main(
            source=args.source,
            model=args.model,
            conf=args.conf,
            save=not args.no_save,
        )


if __name__ == "__main__":
    cli()