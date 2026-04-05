import torch
from model_loader import model

def predict_from_features(features):
    with torch.no_grad():
        features = torch.tensor(features).float().unsqueeze(0)
        output = model(features)
        score = torch.sigmoid(output).item()
        label = "High Risk" if score > 0.5 else "Low Risk"
        return score, label