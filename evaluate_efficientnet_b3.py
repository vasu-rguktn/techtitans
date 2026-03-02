# evaluate_efficientnet_test.py all kinds of images including wgan fakes
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# CONFIG
# ==============================
TEST_CSV = "test_auth_final.csv"
MODEL_PATH = "auth_model_best.pt"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# TRANSFORM (same as training)
# ==============================
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==============================
# DATASET
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

test_loader = DataLoader(
    AuthDataset(TEST_CSV),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==============================
# LOAD MODEL
# ==============================
model = timm.create_model("efficientnet_b3", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

best_threshold = checkpoint["best_threshold"]

model = model.to(device)
model.eval()

print("✅ EfficientNet model loaded")
print(f"Using threshold: {best_threshold:.4f}")

# ==============================
# EVALUATION
# ==============================
all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels).flatten()

# Metrics
auc = roc_auc_score(all_labels, all_probs)

preds = (all_probs > best_threshold).astype(int)

accuracy = accuracy_score(all_labels, preds)
precision = precision_score(all_labels, preds)
recall = recall_score(all_labels, preds)
f1 = f1_score(all_labels, preds)
cm = confusion_matrix(all_labels, preds)

print("\n📊 EfficientNet Test Results")
print("--------------------------------")
print(f"ROC-AUC   : {auc:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix")
print(cm)