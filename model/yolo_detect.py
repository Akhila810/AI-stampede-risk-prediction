from ultralytics import YOLO
import cv2
import os
import csv

MODEL_PATH = "detection/yolov8n.pt"
 # lightweight, sufficient
FRAME_DIR = "data/frames"
OUTPUT_CSV = "data/labels/yolo_detections.csv"

model = YOLO(MODEL_PATH)

os.makedirs("data/labels", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video_id",
        "frame_id",
        "person_count",
        "centroids"
    ])

    for video_id in os.listdir(FRAME_DIR):
        video_path = os.path.join(FRAME_DIR, video_id)
        if not os.path.isdir(video_path):
            continue

        for frame_name in sorted(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)

            results = model(frame, classes=[0], verbose=False)

            centroids = []
            person_count = 0

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes.xyxy:
                        x1, y1, x2, y2 = box.tolist()
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        centroids.append((cx, cy))
                        person_count += 1

            writer.writerow([
                video_id,
                frame_name,
                person_count,
                centroids
            ])

print("YOLOv8 detection completed.")
