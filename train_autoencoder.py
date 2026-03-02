
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# ----------------------------
# Dataset
# ----------------------------
class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.tf = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)

# ----------------------------
# Improved Autoencoder Model
# ----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔥 Deeper Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,256,4,2,1), nn.ReLU()
        )

        # 🔥 Deeper Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid()
        )

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# Training block
# ----------------------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = XrayDataset("archive/images/images_normalized")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoencoder().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # 🔥 Use L1 Loss (sharper than MSE)
    loss_fn = nn.L1Loss()

    # 🔥 Train longer
    epochs = 15

    for epoch in range(epochs):
        total_loss = 0

        for imgs in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            loss = loss_fn(out, imgs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "autoencoder.pt")
    print("✅ Improved Autoencoder trained and saved")