import os
import pandas as pd

data = []

# =========================
# REAL IMAGES
# =========================
real_root = r"C:\Users\pooji\Downloads\archive\images\images_normalized"

for file in os.listdir(real_root):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(real_root, file)
        data.append([full_path, 0])

print("Real images collected:", len(data))


# =========================
# DISTORTION FAKE IMAGES
# =========================
fake_root = r"D:\Fake_images"

for subfolder in os.listdir(fake_root):
    subfolder_path = os.path.join(fake_root, subfolder)

    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(subfolder_path, file)
                data.append([full_path, 1])

print("After distortion fakes:", len(data))


# =========================
# GAN FAKE
# =========================
gan_root = r"D:\Desktop\gan_fake_dataset"

for file in os.listdir(gan_root):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(gan_root, file)
        data.append([full_path, 1])

print("After GAN fakes:", len(data))


# =========================
# AUTOENCODER FAKE
# =========================
ae_root = r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1"

for file in os.listdir(ae_root):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(ae_root, file)
        data.append([full_path, 1])

print("After AE fakes:", len(data))


# =========================
# DIFFUSION FAKE
# =========================
diff_root = r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays"

for file in os.listdir(diff_root):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(diff_root, file)
        data.append([full_path, 1])

print("After diffusion fakes:", len(data))


# =========================
# SAVE MASTER CSV
# =========================
df = pd.DataFrame(data, columns=["image", "label"])
df.to_csv("auth_full_dataset.csv", index=False)

print("✅ Master dataset CSV created successfully!")
print("Total images:", len(df))