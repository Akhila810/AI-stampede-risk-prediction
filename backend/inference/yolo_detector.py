from ultralytics import YOLO

# Load YOLOv8 model (people detection)
model = YOLO("yolov8n.pt")  # uses pretrained model

def detect_people(frame):
    """
    Returns list of bounding boxes for people.
    Each box: {x1, y1, x2, y2}
    """
    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    return detections
