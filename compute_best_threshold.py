import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Dataset
# ------------------------------
class AuthDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image).convert("RGB")
        img = self.tf(img)
        label = torch.tensor(row.label).float()
        return img, label


# ------------------------------
# Load validation data
# ------------------------------
val_ds = AuthDataset("val_auth_new.csv")
val_loader = DataLoader(val_ds, batch_size=16)

# ------------------------------
# Load model
# ------------------------------
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features,1)
model.load_state_dict(torch.load("auth_model_new.pt", map_location=device))
model = model.to(device)
model.eval()

# ------------------------------
# Collect probabilities
# ------------------------------
all_probs = []
all_labels = []

with torch.no_grad():
    for x,y in val_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ------------------------------
# Compute ROC
# ------------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# ------------------------------
# Find Best Threshold (Youden's J)
# ------------------------------
J = tpr - fpr
best_idx = np.argmax(J)
best_threshold = thresholds[best_idx]

print("\n🔥 Best Threshold:", best_threshold)
print("ROC AUC:", roc_auc)

# ------------------------------
# Plot ROC Curve
# ------------------------------
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()