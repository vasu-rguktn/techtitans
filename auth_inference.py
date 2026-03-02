import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAME AS TRAINING
tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = timm.create_model("efficientnet_b3", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 1)

checkpoint = torch.load("auth_model_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = checkpoint["best_threshold"]

model = model.to(device)
model.eval()

def authenticity_check(img_path):
    img = Image.open(img_path).convert("RGB")
    img = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    if prob > best_threshold:
        return 1, prob  # Fake
    else:
        return 0, prob  # Real