import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from hybrid.hybrid_model import HybridModel

TEST_CSV = "test_auth_final.csv"
GAN_WEIGHTS = "gan_discriminator.pth"
MODEL_PATH = "hybrid_model_best.pt"

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

test_loader = DataLoader(
    AuthDataset(TEST_CSV),
    batch_size=16
)

model = HybridModel(GAN_WEIGHTS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

auc = roc_auc_score(all_labels, np.array(all_probs).flatten())
acc = accuracy_score(all_labels, np.array(all_probs).flatten() > 0.5)

print("\n📊 Hybrid Test Results")
print("---------------------------")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Accuracy : {acc:.4f}")