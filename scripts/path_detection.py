import os
import sys
import time
from pathlib import Path
import numpy as np

import cv2
from ultralytics import SAM

# models/ lives at the project root, one level up from scripts/
MODELS_DIR = Path(__file__).parent.parent / "models"


def main() -> None:
    # Check if an image path was provided as an argument
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Initializing SAM model...")
    model = SAM(str(MODELS_DIR / "mobile_sam.pt"))

    if image_path:
        # IMAGE MODE
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return

        print(f"Processing image: {image_path}")
        # frame = cv2.imread(image_path)

        # Video Loop:

        frame_count = 0
        frame_analysis_frac = 5
        cap = cv2.VideoCapture(image_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                print("Error: Could not decode image.")
                return

            if frame_count % frame_analysis_frac == 0:
                width = frame.shape[1]
                height = frame.shape[0]
                # ctrdot = [width // 2, height]
                # ctrdots = [ctrdot]
                ctrdots = [[x, height] for x in [2 * width // 5, width // 2, 3 * width // 5]]
                # bbox = [2 * width // 5, height-1, 3 * width // 5, height]

                start_time = time.time()
                results = model.predict(frame, conf=0.25, points=ctrdots)
                end_time = time.time()

                # annotated_frame = results[0].plot()
                annotated_frame = frame.copy()
                print(f" Done ({end_time - start_time:.2f}s)")

                
                if results[0].masks:
                    polygons = results[0].masks.xy
                    
                    if polygons:
                        # Find the polygon with the largest area
                        largest_polygon = max(polygons, key=cv2.contourArea)
                        
                        # Draw the largest polygon in GREEN
                        pts = largest_polygon.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], isClosed=True, 
                                      color=(0, 255, 0), thickness=2)
                cv2.imshow("SAM Segmentation - Image", annotated_frame)

            cv2.waitKey(1)
            frame_count += 1

        cv2.destroyAllWindows()
        print("Press any key in the window to exit.")
