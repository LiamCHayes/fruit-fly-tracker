"""
ResNet50 encoder for pixel-wise segmentation

Uses a Hi-Res head to transform high resolution images to ResNet50 input size
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        x = self.encoder_layers[0](x)
        x = self.encoder_layers[1](x)
        x = self.encoder_layers[2](x)
        x = self.encoder_layers[3](x)

        # Extract features at different stages for skip connections
        extracted_features = []
        x = self.encoder_layers[4](x)
        extracted_features.append(x)
        x = self.encoder_layers[5](x)
        extracted_features.append(x)
        x = self.encoder_layers[6](x)
        extracted_features.append(x)
        x = self.encoder_layers[7](x)

        return x, extracted_features


