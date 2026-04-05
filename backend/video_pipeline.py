import cv2
import numpy as np
import torch
from backend.model_loader import model

class_names = ["Low Risk", "Medium Risk", "High Risk"]

SEQUENCE_LENGTH = 10
FEATURE_DIM = 7


def extract_basic_features(frame):
    """
    TEMPORARY VERSION.
    Replace this with your real YOLO + optical flow logic.
    Must return 7 features per frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Example dummy feature engineering (replace later)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    h, w = gray.shape
    aspect_ratio = w / h

    edge_count = np.sum(cv2.Canny(gray, 100, 200)) / 255

    # Dummy placeholders to reach 7 dims
    f5 = mean_intensity * 0.01
    f6 = std_intensity * 0.01
    f7 = edge_count * 0.0001

    return np.array([
        mean_intensity,
        std_intensity,
        aspect_ratio,
        edge_count,
        f5,
        f6,
        f7
    ], dtype=np.float32)


def build_sequence_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feature_vector = extract_basic_features(frame)
        features.append(feature_vector)

    cap.release()

    if len(features) < SEQUENCE_LENGTH:
        raise ValueError("Video too short to build sequence")

    # Take first 10 frames for now
    sequence = np.array(features[:SEQUENCE_LENGTH])

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise ValueError(f"Invalid sequence shape: {sequence.shape}")

    return sequence


def predict_video(video_path):
    sequence = build_sequence_from_video(video_path)

    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "risk_label": class_names[predicted_class],
        "confidence": round(confidence, 4)
    }