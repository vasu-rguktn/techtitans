import torch
import torch.nn as nn
import timm
from models.discriminator import Discriminator

class HybridModel(nn.Module):
    def __init__(self, gan_weights_path):
        super().__init__()

        # GAN Discriminator
        self.gan_disc = Discriminator()
        self.gan_disc.load_state_dict(
            torch.load(gan_weights_path, map_location="cpu")
        )

        self.gan_disc.classifier = nn.Identity()

        for param in self.gan_disc.parameters():
            param.requires_grad = False

        # Automatically detect GAN feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 128)
            feat = self.gan_disc.features(dummy)
            gan_feature_dim = feat.view(1, -1).shape[1]

        # EfficientNet
        self.effnet = timm.create_model("efficientnet_b3", pretrained=True)
        eff_feature_dim = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Identity()

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(gan_feature_dim + eff_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):

        gan_input = nn.functional.interpolate(
            x, size=(128,128), mode="bilinear"
        )

        gan_features = self.gan_disc.features(gan_input)
        gan_features = gan_features.view(gan_features.size(0), -1)

        eff_features = self.effnet(x)

        combined = torch.cat([gan_features, eff_features], dim=1)

        out = self.classifier(combined)

        return out