"""Decoders for resnet encoder"""

import torch.nn as nn

class ResNet50Decoder(nn.Module):
    def __init__(self, num_classes, encoder):
        super().__init__()
        self.encoder = encoder

        # Decoder layers - progressively upsample features
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x, extracted_features = self.encoder(x)

        # Decoder
        x = self.upconv4(x)
        x = x + extracted_features[2]  # skip connection from layer3
        x = self.conv4(x)
        x = self.upconv3(x)
        x = x + extracted_features[1]  # skip connection from layer2
        x = self.conv3(x)
        x = self.upconv2(x)
        x = x + extracted_features[0]  # skip connection from layer1
        x = self.conv2(x)
        x = self.upconv1(x)
        x = self.conv1(x)

        x = self.upconv0(x)
        x = self.classifier(x)
        return x

