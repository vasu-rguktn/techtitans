import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import os

# ==============================
# CONFIG
# ==============================
TRAIN_CSV = "train_auth_full.csv"
VAL_CSV   = "val_auth_full.csv"
MODEL_SAVE_PATH = "auth_model_best.pt"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
PATIENCE = 4   # Early stopping patience

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# TRANSFORM (EfficientNet-B3 default size = 300)
# ==============================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# DATASET CLASS
# ==============================
class AuthDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image).convert("RGB")
        img = transform(img)
        label = torch.tensor(row.label).float()
        return img, label


train_ds = AuthDataset(TRAIN_CSV)
val_ds   = AuthDataset(VAL_CSV)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ==============================
# MODEL
# ==============================
model = timm.create_model("efficientnet_b3", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(device)

# ==============================
# LOSS (Weighted BCE)
# ==============================
train_df = pd.read_csv(TRAIN_CSV)
real_count = (train_df.label == 0).sum()
fake_count = (train_df.label == 1).sum()

pos_weight = torch.tensor([real_count / fake_count]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ==============================
# TRAINING LOOP
# ==============================
best_auc = 0
patience_counter = 0

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    val_auc = roc_auc_score(all_labels, all_probs)

    # Find best threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")

    # ----- EARLY STOPPING -----
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "best_auc": best_auc
        }, MODEL_SAVE_PATH)

        print("🔥 Best model saved!")

    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("⛔ Early stopping triggered")
            break

print("\n✅ Training Complete")
print(f"Best Validation AUC: {best_auc:.4f}")