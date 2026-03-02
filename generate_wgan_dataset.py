import torch
import torchvision.utils as vutils
import os
from train_wgan_gp_medical import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CONFIG
# -------------------------
LATENT_DIM = 100
NUM_IMAGES = 7000        # change if needed
BATCH_SIZE = 64
OUTPUT_DIR = r"D:\Desktop\gan_fake_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Generator
# -------------------------
G = Generator().to(device)
G.load_state_dict(torch.load("wgan_generator_medical.pth", map_location=device))
G.eval()

print("✅ Generator loaded")

# -------------------------
# Generate Images
# -------------------------
count = 0

with torch.no_grad():
    while count < NUM_IMAGES:

        current_batch = min(BATCH_SIZE, NUM_IMAGES - count)

        noise = torch.randn(current_batch, LATENT_DIM, 1, 1, device=device)
        fake = G(noise)

        # Convert from [-1,1] → [0,1]
        fake = (fake + 1) / 2

        for i in range(current_batch):
            save_path = os.path.join(OUTPUT_DIR, f"gan_fake_{count+i}.png")
            vutils.save_image(fake[i], save_path)

        count += current_batch
        print(f"Generated {count}/{NUM_IMAGES}")

print("🚀 GAN fake dataset generation complete.")