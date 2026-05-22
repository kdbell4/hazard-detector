import cv2
from ultralytics import YOLO

base_model      = YOLO("yolov8m.pt")
pothole_model   = YOLO("runs/detect/train-6/weights/best.pt")
rocks_model     = YOLO("/Users/ashwinkannan/Downloads/rocks_best.pt")
stairs_model    = YOLO("/Users/ashwinkannan/Downloads/stairs_best.pt")
curb_model      = YOLO("/Users/ashwinkannan/Downloads/curb_best.pt")
speedbump_model = YOLO("/Users/ashwinkannan/Downloads/speedbump_best.pt")

HAZARD_IDS = {0,1,2,3,5,7,9,10,11,12,13,14,36}

def run_base_model(model, frame, conf=0.50, color=(0,255,0), annotated=None):
    """Only for base YOLO — filters by HAZARD_IDS"""
    results = model(frame, conf=conf, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in HAZARD_IDS:
            continue
        label      = model.names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f'{label} {conf_score:.0%}',
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

def run_model(model, frame, conf=0.25, color=(255,255,255), annotated=None):
    """For all custom models — detects everything"""
    results = model(frame, conf=conf, verbose=False)
    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        label      = results[0].names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f'{label} {conf_score:.0%}',
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    annotated = run_base_model(base_model,  frame, conf=0.80, color=(0,255,0),   annotated=annotated)
    annotated = run_model(pothole_model,    frame, conf=0.80, color=(0,0,255),   annotated=annotated)
    # annotated = run_model(rocks_model,      frame, conf=0.80, color=(0,165,255), annotated=annotated)
    annotated = run_model(stairs_model,     frame, conf=0.80, color=(255,0,0),   annotated=annotated)
    annotated = run_model(curb_model,       frame, conf=0.80, color=(255,255,0), annotated=annotated)
    annotated = run_model(speedbump_model,  frame, conf=0.80, color=(255,0,255), annotated=annotated)

    cv2.imshow('Hazard Detection', annotated)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
