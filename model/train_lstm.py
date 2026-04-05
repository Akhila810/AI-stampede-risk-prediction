import ast
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from model.lstm_model import RiskLSTM
import random

# --------------------
# Reproducibility
# --------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------
# Config
# --------------------
DATASET_CSV = "data/labels/lstm_sequences.csv"
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001

MODEL_SAVE_PATH = "models/best_model.pth"
os.makedirs("models", exist_ok=True)

# --------------------
# Load Dataset
# --------------------
df = pd.read_csv(DATASET_CSV)

print("Class distribution:")
print(df["risk_label"].value_counts())

class SequenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = df["sequence"].apply(ast.literal_eval).tolist()
        self.labels = df["risk_label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["risk_label"], random_state=42
)

train_loader = DataLoader(SequenceDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SequenceDataset(val_df), batch_size=BATCH_SIZE)

device = torch.device("cpu")
model = RiskLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# --------------------
# Training Loop
# --------------------
for epoch in range(EPOCHS):

    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # DEBUG: Check input shape first time
        if epoch == 0:
            print("Input shape:", x.shape)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total
    train_loss /= len(train_loader)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best model saved")

# --------------------
# Final Evaluation
# --------------------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")