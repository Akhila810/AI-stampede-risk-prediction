import os
import torch
import pandas as pd
import numpy as np

YOLO_CSV = "data/labels/yolo_detections.csv"
FLOW_CSV = "data/labels/optical_flow_features.csv"
GRAPH_DIR = "data/graphs"
META_CSV = "data/metadata.csv"

SEQ_LEN = 10  # time steps per sequence
OUTPUT_CSV = "data/labels/lstm_sequences.csv"

yolo_df = pd.read_csv(YOLO_CSV)
flow_df = pd.read_csv(FLOW_CSV)
meta_df = pd.read_csv(META_CSV, encoding="latin1")

# DEBUG: print columns once
print("Metadata columns BEFORE fix:", meta_df.columns.tolist())

# Clean column names
meta_df.columns = (
    meta_df.columns
    .str.replace('\ufeff', '', regex=False)
    .str.strip()
    .str.lower()
)

# FORCE first column to be video_id (authoritative)
meta_df.rename(columns={meta_df.columns[0]: "video_id"}, inplace=True)

print("Metadata columns AFTER fix:", meta_df.columns.tolist())





rows = []

for video_id in meta_df["video_id"]:
    yolo_v = yolo_df[yolo_df["video_id"] == video_id]
    flow_v = flow_df[flow_df["video_id"] == video_id]
    label = meta_df[meta_df["video_id"] == video_id]["risk_label"].values[0]

    merged = pd.merge(yolo_v, flow_v, on=["video_id", "frame_id"])

    features = []

    for _, row in merged.iterrows():
        graph_file = f"{video_id}_{row['frame_id']}.pt"
        graph_path = os.path.join(GRAPH_DIR, graph_file)

        if not os.path.exists(graph_path):
            continue

        gnn_data = torch.load(graph_path)
        gnn_embed = torch.mean(gnn_data.x, dim=0).numpy()

        vec = [
            row["person_count"],
            row["mean_magnitude"],
            row["flow_variance"],
            row["direction_entropy"],
        ] + gnn_embed.tolist()

        features.append(vec)

    for i in range(0, len(features) - SEQ_LEN):
        seq = features[i:i + SEQ_LEN]
        rows.append([video_id, seq, label])

pd.DataFrame(rows, columns=["video_id", "sequence", "risk_label"]).to_csv(
    OUTPUT_CSV, index=False
)

print("LSTM sequence dataset created.")
