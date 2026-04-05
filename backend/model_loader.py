import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.lstm_model import RiskLSTM

device = torch.device("cpu")

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = RiskLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()