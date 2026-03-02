import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from hybrid.hybrid_model import HybridModel

# CONFIG
TRAIN_CSV = "train_auth_full.csv"
VAL_CSV   = "val_auth_full.csv"
GAN_WEIGHTS = "gan_discriminator.pth"
SAVE_PATH = "hybrid_model_best.pt"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

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

train_loader = DataLoader(
    AuthDataset(TRAIN_CSV),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    AuthDataset(VAL_CSV),
    batch_size=BATCH_SIZE
)

model = HybridModel(GAN_WEIGHTS).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

best_auc = 0

for epoch in range(EPOCHS):

    model.train()

    for x, y in train_loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    val_auc = roc_auc_score(
        np.array(all_labels),
        np.array(all_probs).flatten()
    )

    print(f"Epoch {epoch+1} - Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), SAVE_PATH)
        print("🔥 Hybrid model saved")

print("✅ Hybrid Training Complete")