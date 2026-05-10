import cv2
from ultralytics import YOLO
import pygame

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    results = model(frame)
    alert = False

    for r in results:
        for box in r.boxes:
            # Fixed index access here [0]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords

            box_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            area_ratio = box_area / frame_area

            # Hazard conditions
            if area_ratio > 0.25 and y2 > (h * 0.6):
                alert = True
                break 

    if alert and not pygame.mixer.get_busy():
        alert_sound.play() # Fixed: removed the dot between alert and sound

    # Use results[0] to get the first frame's plotted image
    cv2.imshow("YOLO Alert System", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
