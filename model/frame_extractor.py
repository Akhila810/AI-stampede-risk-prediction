import cv2
import os

VIDEO_DIR = "data/processed_videos"
FRAME_DIR = "data/frames"

os.makedirs(FRAME_DIR, exist_ok=True)

for video_name in os.listdir(VIDEO_DIR):
    video_path = os.path.join(VIDEO_DIR, video_name)
    video_id = os.path.splitext(video_name)[0]

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    video_frame_dir = os.path.join(FRAME_DIR, video_id)
    os.makedirs(video_frame_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(
            video_frame_dir, f"frame_{frame_idx:05d}.jpg"
        )
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()

print("Frame extraction completed.")
