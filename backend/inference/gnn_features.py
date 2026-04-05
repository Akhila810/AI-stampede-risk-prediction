import numpy as np

def build_graph_features(detections, frame_shape):
    h, w = frame_shape[:2]

    centers = []

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)

    if len(centers) == 0:
        return [0.0] * 5

    # 🔥 adjacency (distance-based)
    dist_matrix = np.linalg.norm(
        centers[:, None] - centers[None, :],
        axis=2
    )

    threshold = 50  # tune this

    adjacency = (dist_matrix < threshold).astype(int)

    # remove self connections
    np.fill_diagonal(adjacency, 0)

    # 🔥 graph metrics
    num_edges = np.sum(adjacency) / 2
    avg_degree = np.mean(np.sum(adjacency, axis=1))
    density = num_edges / (len(centers)**2 + 1e-6)

    return [
        len(centers),
        num_edges,
        avg_degree,
        density,
        np.std(dist_matrix)
    ]