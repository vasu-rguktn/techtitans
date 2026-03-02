import os
import pandas as pd
import random

REAL_DIR = r"C:\Users\pooji\Downloads\archive\images\images_normalized"
FAKE_DIR = r"D:\Desktop\gan_fake_dataset"

OUTPUT_TRAIN = "train_gan_auth.csv"
OUTPUT_VAL = "val_gan_auth.csv"

# -------------------------
# Collect images
# -------------------------
real_images = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR)]
fake_images = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR)]

# Balance dataset
min_count = min(len(real_images), len(fake_images))

real_images = random.sample(real_images, min_count)
fake_images = random.sample(fake_images, min_count)

data = []

for img in real_images:
    data.append([img, 0])

for img in fake_images:
    data.append([img, 1])

random.shuffle(data)

# 80-20 split
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

pd.DataFrame(train_data, columns=["image", "label"]).to_csv(OUTPUT_TRAIN, index=False)
pd.DataFrame(val_data, columns=["image", "label"]).to_csv(OUTPUT_VAL, index=False)

print("✅ GAN authenticity dataset created")
print("Train:", len(train_data))
print("Val:", len(val_data))