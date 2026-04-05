import cv2
import numpy as np


def compute_flow(prev_gray, gray):
    """
    Computes optical flow magnitude between two grayscale frames
    Returns magnitude matrix
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag
