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

    results = model(frame) 

    for r in results:
        if len(r.boxes) > 0:
            if not pygame.mixer.get_busy():
                alert_sound.play()

    cv2.imshow("YOLO Alert System", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()