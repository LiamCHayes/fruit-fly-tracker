"""Base case for fly detection. Unet trained on synthetic data."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
from utilities import SyntheticData

# Hyperparameters
epochs = 10
batch_size = 1
device = 'cuda'

# Set up for training
model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
        )
model.to(device)

preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
optimizer = optim.Adam(model.parameters(), lr= 0.0001)

image_dir = "../data_generation/dataset/train/images/"
label_dir = "../data_generation/dataset/train/masks/"
background_dir = "../data_generation/dataset/backgrounds/"
dataset = SyntheticData(image_dir, label_dir, background_dir)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

# Training loop
model.train()
epoch_losses = [0.0 for _ in range(epochs)]
for epoch in range(epochs):
    running_loss = 0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
        images = images.permute(0, 2, 3, 1)
        labels = labels.permute(0, 2, 3, 1).to(device)
        processed_images = preprocess_input(images)
        processed_images = processed_images.permute(0, 3, 1, 2).float().to(device)

        optimizer.zero_grad()
        outputs = model(processed_images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item()

    # Record loss
    running_loss /= len(dataloader)
    epoch_losses[epoch] = running_loss

    # Save a forward pass of real data TODO

# make dataframe of losses and save
training_metrics = pl.DataFrame({
    "epoch_loss": epoch_losses
    })
training_metrics.write_csv("training_metrics.csv")
