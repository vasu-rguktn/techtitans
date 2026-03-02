import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Folder paths
# -----------------------------
REAL_DIR = "archive/images/images_normalized"
FAKE_AUTO_DIR = "fake_autoencoder1"
FAKE_DIFF_DIR = "fake_xrays"

OUTPUT_TRAIN = "train_auth_full.csv"
OUTPUT_VAL = "val_auth_full.csv"

# -----------------------------
# Collect image paths
# -----------------------------
real_images = [os.path.join(REAL_DIR, f)
               for f in os.listdir(REAL_DIR)]

fake_auto_images = [os.path.join(FAKE_AUTO_DIR, f)
                    for f in os.listdir(FAKE_AUTO_DIR)]

fake_diff_images = [os.path.join(FAKE_DIFF_DIR, f)
                    for f in os.listdir(FAKE_DIFF_DIR)]

# Label:
# Real = 0
# Fake = 1

data = []

for path in real_images:
    data.append({"image": path, "label": 0})

for path in fake_auto_images:
    data.append({"image": path, "label": 1})

for path in fake_diff_images:
    data.append({"image": path, "label": 1})

df = pd.DataFrame(data)

# -----------------------------
# Balance dataset
# -----------------------------
real_df = df[df.label == 0]
fake_df = df[df.label == 1]

min_size = min(len(real_df), len(fake_df))

real_df = real_df.sample(min_size, random_state=42)
fake_df = fake_df.sample(min_size, random_state=42)

balanced_df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42)

# -----------------------------
# Train / Validation split
# -----------------------------
train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df["label"],
    random_state=42
)

train_df.to_csv(OUTPUT_TRAIN, index=False)
val_df.to_csv(OUTPUT_VAL, index=False)

print("✅ Dataset created")
print("Train size:", len(train_df))
print("Val size:", len(val_df))