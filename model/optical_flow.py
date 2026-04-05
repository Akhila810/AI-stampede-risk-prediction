import cv2
import os
import csv
import numpy as np

FRAME_DIR = "data/frames"
OUTPUT_CSV = "data/labels/optical_flow_features.csv"

os.makedirs("data/labels", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video_id",
        "frame_id",
        "mean_magnitude",
        "flow_variance",
        "direction_entropy"
    ])

    for video_id in os.listdir(FRAME_DIR):
        video_path = os.path.join(FRAME_DIR, video_id)
        if not os.path.isdir(video_path):
            continue

        frame_files = sorted(os.listdir(video_path))
        prev_gray = None

        for frame_name in frame_files:
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            mean_mag = np.mean(mag)
            var_mag = np.var(mag)

            hist, _ = np.histogram(ang, bins=16, range=(0, 2*np.pi), density=True)
            entropy = -np.sum(hist * np.log(hist + 1e-6))

            writer.writerow([
                video_id,
                frame_name,
                round(mean_mag, 5),
                round(var_mag, 5),
                round(entropy, 5)
            ])

            prev_gray = gray

print("Optical flow feature extraction completed.")
