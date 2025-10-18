"""Base case for fly detection. Unet trained on synthetic data."""

import torch
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import argparse
from utilities import SyntheticData
from visual_test_model import test_on_selected_images

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", help="path to the directory of images (end with /)")
parser.add_argument("-l", "--labels", help="path to the directory of labels (end with /)")
parser.add_argument("-b", "--backgrounds", help="path to the directory of backgrounds (end with /)")
args = parser.parse_args()

# Hyperparameters
epochs = 10
batch_size = 1
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

image_dir = args.images
label_dir = args.labels
background_dir = args.backgrounds
dataset = SyntheticData(image_dir, label_dir, background_dir)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

def full_transform(img):
    """
    Full transform that the image goes through from PIL Image to model input
    Used to test on the real data
    """
    img.resize((3072, 2048)) # Dimensions are flipped because img is a PIL.Image object
    img_transform = dataset.img_transform
    img = img_transform(img).unsqueeze(0)
    img = img.permute(0, 2, 3, 1)
    img = preprocess_input(img)
    img = img.permute(0, 3, 1, 2).float().to(device)
    return img

# Training loop
model.train()
epoch_losses = [0.0 for _ in range(epochs)]
for epoch in range(epochs):
    running_loss = 0
    print(f"Epoch {epoch}")

    for images, labels in dataloader:
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
    print(f"Loss: {running_loss}\n")

    # Save a forward pass of real data
    test_on_selected_images(model, full_transform, f"unet_{epoch}")

# make dataframe of losses and save
training_metrics = pl.DataFrame({
    "epoch_loss": epoch_losses
    })
training_metrics.write_csv("training_metrics.csv")
