from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, os
from tqdm import tqdm

IMG_DIR = r"C:/Users/pooji/Desktop/majoprojec t/fake_autoencoder1"
OUT_DIR = "fake_reports_autoencoder1"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

def format_report(caption):
    return f"""
FINDINGS:
{caption}. Cardiomediastinal silhouette is within normal limits.
No acute pleural effusion or pneumothorax detected.

IMPRESSION:
Synthetic radiographic appearance. Clinical correlation recommended.
""".strip()

for img_name in tqdm(os.listdir(IMG_DIR)):
    try:
        img_path = os.path.join(IMG_DIR, img_name)

        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_length=60)
        caption = processor.decode(out[0], skip_special_tokens=True)

        report = format_report(caption)

        with open(os.path.join(OUT_DIR, img_name.replace(".png",".txt")), "w") as f:
            f.write(report)

    except Exception as e:
        print("Skipping:", img_name, e)

print("✅ Fake reports generated.")
