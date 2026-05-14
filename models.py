import cv2
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

pothole_model = YOLO("runs/detect/train-6/weights/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    pothole_results = pothole_model(frame, conf=0.25)
    results = model(frame,conf=0.5)

    annotated_frame = results[0].plot()

    for box in pothole_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, 'Pothole', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
