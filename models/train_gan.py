
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from models.custom_dataset_class_gan import RealDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = RealDataset("C:\\Users\\pooji\\Desktop\\majoprojec t\\archive\\images\\images_normalized")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Models
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

z_dim = 100
EPOCHS = 12

for epoch in range(EPOCHS):
    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0

    for real_images in loader:

        real_images = real_images.to(device)
        real_images = real_images + 0.05 * torch.randn_like(real_images)
        real_images = torch.clamp(real_images, -1, 1)
        batch_size = real_images.size(0)

        # real_labels = torch.ones(batch_size, 1).to(device)
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_images = G(noise)

        D_real = D(real_images)
        D_fake = D(fake_images.detach())

        d_loss = criterion(D_real, real_labels) + \
                 criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # # Train Generator
        # D_fake = D(fake_images)
        # g_loss = criterion(D_fake, real_labels)

        # optimizer_G.zero_grad()
        # g_loss.backward()
        # optimizer_G.step()
        # -----------------------
        # Train Generator (Twice)
        # -----------------------
        for _ in range(3):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_images = G(noise)

            D_fake = D(fake_images)
            g_loss = criterion(D_fake, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        num_batches += 1

    print(f"Epoch [{epoch+1}/{EPOCHS}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"D_loss: {total_d_loss/num_batches:.4f} "
          f"G_loss: {total_g_loss/num_batches:.4f}")

# Save models
torch.save(D.state_dict(), "gan_discriminator.pth")
torch.save(G.state_dict(), "gan_generator.pth")

print("✅ GAN training complete and models saved.")