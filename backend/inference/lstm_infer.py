import torch
import torch.nn as nn
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size=68, hidden_size=64, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


class LSTMInfer:
    def __init__(self, model_path):
        self.device = "cpu"

        # 🔥 EXACT architecture used during training
        self.model = LSTMModel(
            input_size=68,
            hidden_size=64,
            num_layers=1,
            num_classes=3
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, sequence_30x68):
        """
        sequence_30x68: numpy array (30, 68)
        """
        if sequence_30x68.shape != (30, 68):
            raise ValueError(f"Expected (30,68), got {sequence_30x68.shape}")

        with torch.no_grad():
            x = torch.tensor(sequence_30x68, dtype=torch.float32)
            x = x.unsqueeze(0)  # (1, 30, 68)

            logits = self.model(x)
            pred = torch.argmax(logits, dim=1).item()

        return pred
