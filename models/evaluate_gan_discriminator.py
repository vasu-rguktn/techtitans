import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Config
# -----------------------------
TEST_CSV = "test_auth_final.csv"   # CSV with real + fake images
BATCH_SIZE = 32

# -----------------------------
# Transform (MUST match GAN training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# Dataset
# -----------------------------
class TestDataset(Dataset):
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

test_loader = DataLoader(TestDataset(TEST_CSV),
                         batch_size=BATCH_SIZE,
                         shuffle=False)

# -----------------------------
# Load Model
# -----------------------------
model = Discriminator().to(device)
model.load_state_dict(torch.load("gan_discriminator.pth", map_location=device))
model.eval()

print("✅ GAN Discriminator Loaded")

# -----------------------------
# Testing
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = torch.sigmoid(model(images))
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels)

# Metrics
auc = roc_auc_score(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds > 0.5)

print("ROC-AUC:", round(auc, 4))
print("Accuracy:", round(acc, 4))