import torch, timm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
#for manual debugging porpose only, not for evaluation
device = "cuda"

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features,1)
model.load_state_dict(torch.load("auth_model_new.pt"))
model = model.to(device)
model.eval()

# img = Image.open(r"C:\Users\pooji\Desktop\majoprojec t\pictures\64_IM-2218-4004.dcm.png").convert("RGB")
img = Image.open(r"C:\Users\pooji\Desktop\majoprojec t\fake_autoencoder1\1_IM-0001-4001.dcm.png").convert("RGB")
img = tf(img).unsqueeze(0).to(device)
img1=Image.open(r"C:\Users\pooji\Desktop\majoprojec t\fake_xrays\1_IM-0001-4001.dcm.png").convert("RGB")
img1 = tf(img1).unsqueeze(0).to(device)
prob = torch.sigmoid(model(img)).item()
prob1 = torch.sigmoid(model(img1)).item()

label = "fake" if prob > 0.5 else "real"     
label1 = "fake" if prob1 > 0.5 else "real"

print("Probability:", prob)
print("Predicted label:", label)

print("Probability:", prob1)
print("Predicted label:", label1)
