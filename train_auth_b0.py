
import torch
import torch.nn as nn
import timm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class AuthDataset(Dataset):
    def __init__(self, csv, train=True):
        self.df = pd.read_csv(csv)

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
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


# --------------------------------------------------
# Load Data
# --------------------------------------------------
train_ds = AuthDataset("train_auth_full.csv", train=True)
val_ds   = AuthDataset("val_auth_full.csv", train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)


# --------------------------------------------------
# Model
# --------------------------------------------------
model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 1)
model = model.to(device)


# --------------------------------------------------
# Loss with class imbalance handling
# --------------------------------------------------
pos_weight = torch.tensor([2.0]).to(device)  # adjust if needed
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)


# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 15

for epoch in range(epochs):

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# --------------------------------------------------
# Save Model
# --------------------------------------------------
torch.save(model.state_dict(), "auth_model_new.pt")
print("\n✅ Authenticity model saved as auth_model_new.pt")