import cv2
import numpy as np
import time
from collections import deque

from inference.yolo_detector import detect_people
from inference.optical_flow import compute_flow
from inference.temporal_buffer import TemporalBuffer
from inference.lstm_infer import LSTMInfer
from inference.gnn_features import build_graph_features
from socket_instance import socketio

# -------------------------------
# INITIALIZATION
# -------------------------------

temporal = TemporalBuffer(sequence_length=10)
import os

MODEL_PATH = "models/lstm_model.pt"

if os.path.exists(MODEL_PATH):
    lstm = LSTMInfer(MODEL_PATH)
else:
    print("⚠️ Model not found. Running without LSTM.")
    lstm = None

# 🔥 HISTORY STORAGE
history_data = []

risk_history = deque(maxlen=7)

# -------------------------------
# FEATURE VECTOR (58D)
# -------------------------------
print("Processing started")
def build_feature_vector(frame, detections, flow_mag):
    h, w = frame.shape[:2]

    areas, widths, heights, confs = [], [], [], []

    for d in detections:
        if isinstance(d, dict):
            x1 = float(d["x1"])
            y1 = float(d["y1"])
            x2 = float(d["x2"])
            y2 = float(d["y2"])
            conf = float(d.get("conf", 1.0))
        else:
            if len(d) == 5:
                x1, y1, x2, y2, conf = d
            else:
                x1, y1, x2, y2 = d
                conf = 1.0

            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            conf = float(conf)

        widths.append(x2 - x1)
        heights.append(y2 - y1)
        areas.append((x2 - x1) * (y2 - y1))
        confs.append(conf)

    people_count = len(detections)

    if people_count > 0:
        crowd_feats = [
            people_count,
            float(np.mean(areas)),
            float(np.std(areas)),
            float(np.mean(widths)),
            float(np.mean(heights)),
            people_count / (h * w),
            float(np.mean(confs)),
            float(np.max(confs)),
        ]
    else:
        crowd_feats = [0.0] * 8

    mag_flat = flow_mag.flatten()

    if mag_flat.size == 0:
        flow_feats = [0.0] * 50
    else:
        hist, _ = np.histogram(
            mag_flat,
            bins=50,
            range=(0, np.max(mag_flat) + 1e-6)
        )
        hist = hist / (np.sum(hist) + 1e-6)
        flow_feats = hist.tolist()

    return crowd_feats + flow_feats

# -------------------------------
# VIDEO PROCESSING
# -------------------------------

def process_video(video_path):
    global history_data

    cap = cv2.VideoCapture(video_path)

    prev_gray = None
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detect_people(frame)
        people_count = len(detections)

        if prev_gray is None:
            prev_gray = gray
            continue

        # Optical Flow
        flow_mag = compute_flow(prev_gray, gray)
        flow_intensity = float(np.mean(flow_mag))

        # Feature vector
        from inference.gnn_features import build_graph_features

        base_features = build_feature_vector(frame, detections, flow_mag)
        graph_features = build_graph_features(detections, frame.shape)

        
        
        feature_vector = base_features + graph_features
        temporal.add(feature_vector)

        if not temporal.is_ready():
            prev_gray = gray
            frame_id += 1
            continue

        # LSTM input
        sequence = temporal.get_sequence()
        pad = np.zeros((sequence.shape[0], 10))
        sequence_68 = np.hstack([sequence, pad])

        # -------------------------------
        # RISK LOGIC
        # -------------------------------

        if people_count <= 2:
            base_risk = "LOW"
        elif people_count <= 4:
            base_risk = "MEDIUM"
        else:
            base_risk = "HIGH"

        if flow_intensity > 2.5:
            if base_risk == "LOW":
                base_risk = "MEDIUM"
            elif base_risk == "MEDIUM":
                base_risk = "HIGH"

        try:
            risk_id = lstm.predict(sequence_68)
            lstm_risk = ["LOW", "MEDIUM", "HIGH"][risk_id]
        except:
            lstm_risk = base_risk

        if base_risk == "HIGH":
            current_risk = "HIGH"
        elif base_risk == "MEDIUM":
            current_risk = "HIGH" if lstm_risk == "HIGH" else "MEDIUM"
        else:
            current_risk = lstm_risk

        # -------------------------------
        # SMOOTHING
        # -------------------------------
        risk_history.append(current_risk)
        risk_label = max(set(risk_history), key=risk_history.count)

        # -------------------------------
        # 🔥 STORE HISTORY (FIX)
        # -------------------------------
        history_data.append({
            "time": time.strftime("%H:%M:%S"),
            "risk": risk_label,
            "people": people_count,
            "flow": round(flow_intensity, 2)
        })

        # keep only last 100 entries
        history_data[:] = history_data[-100:]

        # -------------------------------
        # EMIT TO FRONTEND
        # -------------------------------
        socketio.emit("frame_update", {
            "risk": risk_label,
            "detections": detections,
        })

        # Debug
        print(
            f"[FRAME {frame_id}] "
            f"People={people_count} "
            f"Flow={flow_intensity:.2f} "
            f"Final={risk_label}"
        )

        prev_gray = gray
        frame_id += 1

    cap.release()