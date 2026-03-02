import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from tqdm import tqdm

REAL_DIR = "archive/images/images_normalized"
FAKE_DIR = "fake_xrays_new1"

os.makedirs(FAKE_DIR, exist_ok=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_attention_slicing()

prompt = "realistic chest x-ray, grayscale medical radiograph, lungs visible, hospital scan, high detail"

# for img_name in tqdm(os.listdir(REAL_DIR)):
image_list = os.listdir(REAL_DIR)[:5]

for img_name in tqdm(image_list):

    try:
        img = Image.open(os.path.join(REAL_DIR, img_name)).convert("RGB").resize((512,512))

        fake = pipe(
            prompt=prompt,
            image=img,
            strength=0.85,
            guidance_scale=4.5
        ).images[0]

        fake.save(os.path.join(FAKE_DIR, img_name))

    except Exception as e:
        print("Skipping:", img_name, e)

print("✅ Fake X-rays generated.")
