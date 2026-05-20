import csv
import os
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import pygame

# --- 1. Audio & Logging Setup ---
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")

#LOG_FILE_PATH = "hazard_log.csv"
# Force the CSV to save in the exact same folder as this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "hazard_log.csv")
print(f"Saving data directly to: {LOG_FILE_PATH}")

ALERT_COOLDOWN_SECONDS = 2.0
last_alert_time = 0.0

if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "object_type", "area_ratio", "in_path"])

# --- 2. Model & Camera Initialization ---
print("Loading YOLO and SAM Models (This can take a moment)...")
yolo_model = YOLO("yolov8n.pt")
sam_model = SAM("models/mobile_sam.pt")

# RESET TO STANDARD WEBCAM (Index 0). Change to 1 if using an external plug-in camera.
cap = cv2.VideoCapture(0)

frame_count = 0
frame_analysis_frac = 5  # Optimize: Only calculate road path every 5 frames
path_polygon = None      # Stores the last calculated path

print("Initialization complete! Starting webcam feed...")

# --- 3. Main Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    h, w, _ = frame.shape
    current_time = time.time()
    
    alert_triggered_this_frame = False
    hazards_to_log = []
    annotated_frame = frame.copy()

    # --- PART A: Optimized SAM Path Segmentation ---
    # Only recalculate the path boundary every 5 frames to prevent frame freezing/lag
    if frame_count % frame_analysis_frac == 0:
        ctrdots = [[x, h] for x in [2 * w // 5, w // 2, 3 * w // 5]]
        sam_results = sam_model.predict(frame, conf=0.25, points=ctrdots, verbose=False)
        
        # FIX: Added [0] index to handle the results list correctly
        if sam_results[0].masks and sam_results[0].masks.xy:
            polygons = sam_results[0].masks.xy
            path_polygon = max(polygons, key=cv2.contourArea)

    # Always draw the road path outline if we have a valid segmentation history
    if path_polygon is not None:
        pts = path_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 100), thickness=2)

    # --- PART B: YOLO Object Detection ---
    yolo_results = yolo_model(frame, imgsz=320, verbose=False)

    # FIX: Added [0] index to drill into the first frame's results object
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Fixed: box.xyxy needs [0] index too
        label = yolo_model.names[int(box.cls)]

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / (w * h)

        # Apply proximity rules
        if label == "person":
            too_close = area_ratio > 0.30
        else:
            too_close = area_ratio > 0.08

        # Dynamic Path intersection logic
        bottom_center_pt = (int((x1 + x2) / 2), y2)
        in_path = False
        if path_polygon is not None:
            is_inside = cv2.pointPolygonTest(path_polygon, bottom_center_pt, False)
            in_path = is_inside >= 0

        low = y2 > (h * 0.6)

        # Hazard detection verification
        if too_close and in_path and low:
            alert_triggered_this_frame = True
            color = (0, 0, 255)  # Red for threats
            display_label = f"HAZARD: {label}"
            hazards_to_log.append((label, area_ratio, in_path))
        else:
            color = (255, 0, 0)  # Blue for safe background detections
            display_label = label

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, display_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- PART C: Alerts & Logging Executions ---
    if alert_triggered_this_frame:
        if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
            if not pygame.mixer.get_busy():
                alert_sound.play()
                last_alert_time = current_time

        cv2.putText(annotated_frame, "WARNING: OBSTACLE AHEAD", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            for obj_type, ratio, path_status in hazards_to_log:
                writer.writerow([timestamp_str, obj_type, f"{ratio:.4f}", path_status])

    # Display window update and frame increment tracking
    cv2.imshow("Advanced Scooter Hazard Detection", annotated_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
