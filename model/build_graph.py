import ast
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

YOLO_CSV = "data/labels/yolo_detections.csv"
GRAPH_DIR = "data/graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

df = pd.read_csv(YOLO_CSV)

for (video_id, frame_id), group in df.groupby(["video_id", "frame_id"]):

    centroids = ast.literal_eval(group.iloc[0]["centroids"])

    if len(centroids) < 2:
        continue

    coords = torch.tensor(centroids, dtype=torch.float)

    # Normalize coordinates
    coords = coords / torch.tensor([640, 360])

    nbrs = NearestNeighbors(radius=0.08).fit(coords)
    edges = nbrs.radius_neighbors(coords, return_distance=False)

    edge_index = []
    for i, neighbors in enumerate(edges):
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])

    if len(edge_index) == 0:
        continue

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Node features: x, y, local_density
    local_density = torch.tensor(
        [len(n) for n in edges], dtype=torch.float
    ).unsqueeze(1)

    x = torch.cat([coords, local_density], dim=1)

    data = Data(x=x, edge_index=edge_index)

    torch.save(
        data,
        os.path.join(GRAPH_DIR, f"{video_id}_{frame_id}.pt")
    )

print("Graph construction completed.")
