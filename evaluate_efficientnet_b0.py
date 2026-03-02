import torch, timm, pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

device = "cuda"
#--------FOR ACCURACY EVALUATION--------#
class AuthDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image).convert("RGB")
        img = self.tf(img)
        label = row.label
        return img, label

val_ds = AuthDataset("val_auth.csv")
val_loader = DataLoader(val_ds, batch_size=16)

model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features,1)
model.load_state_dict(torch.load("auth_model.pt"))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x,y in val_loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).cpu()
        pred = (prob > 0.5).int().squeeze()

        y_true.extend(y.numpy())
        y_pred.extend(pred.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
