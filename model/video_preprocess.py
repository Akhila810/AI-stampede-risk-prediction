import cv2
import os

RAW_VIDEO_DIR = "data/raw_videos"
PROCESSED_VIDEO_DIR = "data/processed_videos"

TARGET_FPS = 10
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)

for video_name in os.listdir(RAW_VIDEO_DIR):
    if not video_name.endswith((".mp4", ".avi", ".mov")):
        continue

    input_path = os.path.join(RAW_VIDEO_DIR, video_name)
    output_path = os.path.join(PROCESSED_VIDEO_DIR, video_name)

    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        TARGET_FPS,
        (TARGET_WIDTH, TARGET_HEIGHT)
    )

    frame_interval = int(original_fps / TARGET_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

print("Video preprocessing completed.")
