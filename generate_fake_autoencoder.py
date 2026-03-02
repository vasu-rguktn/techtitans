import torch
from PIL import Image
import os
from torchvision import transforms
from train_autoencoder import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load trained autoencoder
# ----------------------------
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pt", map_location=device))
model.eval()

# ----------------------------
# Transform
# ----------------------------
tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

REAL_DIR = "archive/images/images_normalized"
FAKE_DIR = "fake_autoencoder1"
os.makedirs(FAKE_DIR, exist_ok=True)

image_list = os.listdir(REAL_DIR)

for img_name in image_list:
    img_path = os.path.join(REAL_DIR, img_name)

    img = Image.open(img_path).convert("RGB")
    img_t = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        #  Encode
        latent = model.encoder(img_t)

        #  Add latent perturbation (better than output noise)
        latent = latent + 0.05 * torch.randn_like(latent)

        #  Decode
        fake = model.decoder(latent)
        fake = fake.clamp(0,1)

    save_img = transforms.ToPILImage()(fake.squeeze().cpu())
    save_img.save(os.path.join(FAKE_DIR, img_name))

print(" realistic fake X-rays generated (no retraining).")