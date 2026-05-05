from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam (try 0 or 1 depending on your setup)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Run YOLO detection
    results = model(frame, imgsz=320)

    alert = False

    for box in results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get label
        cls = int(box.cls[0])
        label = model.names[cls]

        # Compute box area
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        frame_area = frame_width * frame_height
        area_ratio = box_area / frame_area

        #  Class-specific thresholds
        if label == "person":
            too_close = area_ratio > 0.30
        else:
            too_close = area_ratio > 0.08

        # Path logic (center of frame)
        object_center = (x1 + x2) / 2
        frame_center = frame_width / 2
        in_path = abs(object_center - frame_center) < frame_width * 0.25

        # Bottom-of-screen check (closer to scooter)
        low = y2 > frame_height * 0.6

        # Final hazard decision
        if too_close and in_path and low:
            alert = True
            color = (0, 0, 255)  # red
            display_label = f"HAZARD: {label}"
        else:
            color = (0, 255, 0)  # green
            display_label = label

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        cv2.putText(frame, display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show global warning
    if alert:
        cv2.putText(frame, "WARNING: OBSTACLE AHEAD", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display frame
    cv2.imshow("Scooter Hazard Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()