import os
import cv2
from tqdm import tqdm

INPUT_DIR = r"C:\Users\pooji\Downloads\archive\images\images_normalized"
OUTPUT_DIR = r"D:\Desktop\gan_real_128"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = os.listdir(INPUT_DIR)

for f in tqdm(files):
    path = os.path.join(INPUT_DIR, f)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (128,128))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f), img)

print("✅ Preprocessing complete.")