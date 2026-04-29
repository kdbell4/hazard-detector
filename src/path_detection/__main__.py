import cv2
from ultralytics import SAM
import time
import sys
import os

def main() -> None:
    # Check if an image path was provided as an argument
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("Initializing SAM model...")
    model = SAM("models/mobile_sam.pt")

    if image_path:
        # IMAGE MODE
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return

        print(f"Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not decode image.")
            return

        width = frame.shape[1]
        height = frame.shape[0]
        ctrdot = [width // 2, height]
        
        print("Running inference...", end="", flush=True)
        start_time = time.time()
        results = model.predict(frame, conf=0.25, points=[ctrdot])
        end_time = time.time()
        
        annotated_frame = results[0].plot()
        print(f" Done ({end_time - start_time:.2f}s)")

        cv2.imshow('SAM Segmentation - Image', annotated_frame)
        print("Press any key in the window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
