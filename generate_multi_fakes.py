import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
REAL_FOLDER = r"C:\Users\pooji\Downloads\archive\images\images_normalized"
OUTPUT_ROOT = r"D:\Fake_images"

REAL_IMAGES = os.listdir(REAL_FOLDER)

TOTAL_REAL = len(REAL_IMAGES)
FAKE_PER_TYPE = TOTAL_REAL // 9   # Balanced distribution

fake_types = [
    "noise", "blur", "jpeg", "contrast",
    "sharp", "resize", "gamma", "elastic", "combo"
]

print(f"Total real images: {TOTAL_REAL}")
print(f"Generating {FAKE_PER_TYPE} fakes per type (~balanced)")

# -----------------------------
# Create folders
# -----------------------------
for ftype in fake_types:
    os.makedirs(os.path.join(OUTPUT_ROOT, ftype), exist_ok=True)


# -----------------------------
# Fake Functions
# -----------------------------
def add_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(img):
    return cv2.GaussianBlur(img, (7, 7), 0)


def add_jpeg(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


def adjust_contrast(img):
    alpha = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def sharpen(img):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)


def resize_artifact(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//2, h//2))
    return cv2.resize(small, (w, h))


def gamma_correction(img):
    gamma = random.uniform(0.5, 1.5)
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def elastic_distortion(img):
    h, w = img.shape[:2]
    dx = (np.random.rand(h, w) * 2 - 1) * 3
    dy = (np.random.rand(h, w) * 2 - 1) * 3

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def combo_attack(img):
    img = add_noise(img)
    img = add_blur(img)
    img = adjust_contrast(img)
    return img


fake_functions = {
    "noise": add_noise,
    "blur": add_blur,
    "jpeg": add_jpeg,
    "contrast": adjust_contrast,
    "sharp": sharpen,
    "resize": resize_artifact,
    "gamma": gamma_correction,
    "elastic": elastic_distortion,
    "combo": combo_attack
}


# -----------------------------
# Generate Balanced Fakes
# -----------------------------
for ftype in fake_types:

    selected_images = random.sample(REAL_IMAGES, FAKE_PER_TYPE)

    print(f"\nGenerating {ftype} fakes...")

    for img_name in tqdm(selected_images):

        img_path = os.path.join(REAL_FOLDER, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        fake_img = fake_functions[ftype](img)

        save_path = os.path.join(OUTPUT_ROOT, ftype, img_name)
        cv2.imwrite(save_path, fake_img)

print("\n✅ Balanced multi-type fake dataset generated successfully.")