import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Hyperparameters
# ---------------------
BATCH_SIZE = 32
EPOCHS = 60
LATENT_DIM = 100
LR = 5e-5
LAMBDA_GP = 5
CRITIC_ITERS = 3

DATASET_PATH = r"D:\Desktop\gan_real_128_parent"

# ---------------------
# Dataset
# ---------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------
# Generator
# ---------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------
# Critic (no sigmoid!)
# ---------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1).mean(1)

# ---------------------
# Gradient Penalty
# ---------------------
def gradient_penalty(real, fake, critic):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    prob_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ==============================
# TRAINING SECTION
# ==============================
if __name__ == "__main__":

    G = Generator().to(device)
    C = Critic().to(device)

    optG = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
    optC = optim.Adam(C.parameters(), lr=LR, betas=(0.0, 0.9))

    for epoch in range(EPOCHS):
        for real, _ in loader:
            real = real.to(device)

            # Train Critic
            for _ in range(CRITIC_ITERS):
                noise = torch.randn(real.size(0), LATENT_DIM, 1, 1, device=device)
                fake = G(noise)

                lossC = -(torch.mean(C(real)) - torch.mean(C(fake)))
                gp = gradient_penalty(real, fake, C)
                lossC += LAMBDA_GP * gp

                optC.zero_grad()
                lossC.backward()
                optC.step()

            # Train Generator
            noise = torch.randn(real.size(0), LATENT_DIM, 1, 1, device=device)
            fake = G(noise)
            lossG = -torch.mean(C(fake))

            optG.zero_grad()
            lossG.backward()
            optG.step()

        print(f"Epoch {epoch+1}/{EPOCHS}  LossC: {lossC.item():.4f}  LossG: {lossG.item():.4f}")

    torch.save(G.state_dict(), "wgan_generator_medical.pth")
    print("✅ WGAN-GP Training Complete")