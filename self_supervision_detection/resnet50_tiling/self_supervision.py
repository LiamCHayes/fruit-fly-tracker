"""Self-supervision training loop for resnet on hi-res backgrounds - Image Completion"""

from itertools import batched
import os
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import polars as pl
from PIL import Image
import matplotlib.pyplot as plt
from encoder import ResNet50Encoder, preprocess, inv_normalize
from decoder import ResNet50Decoder
from utilities import SyntheticData
from tile_picture import ImageTiler

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backgrounds", help="path to the directory of backgrounds (end with /)")
parser.add_argument("-o", "--output", help="path to the output directory (end with /)")
parser.add_argument("-t", "--test", help="path to the test image")
parser.add_argument("--debug", action="store_true", help="make model input 1 tile for fast forward passes")
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

# Hyperparameters
epochs = 10
batch_size = 1
max_h = 128
max_w = 128
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# model training things
encoder = ResNet50Encoder()
model = ResNet50Decoder(3, encoder)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr= 0.0001)
loss_fn = nn.L1Loss()

# data things
dataset = SyntheticData(args.backgrounds, args.backgrounds)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
tiler = ImageTiler()

# load the test image
with Image.open(args.test) as img:
    test_img = dataset.img_transform(img).unsqueeze(0)
test_mask = torch.ones_like(tiled_image)
test_mask[:, :, 100:200, 100:200] = 0

# training loop
epoch_losses = [0.0 for _ in range(epochs)]
for epoch in range(epochs):
    model.train()
    running_loss = 0
    print(f"Epoch {epoch}")

    for images, labels in dataloader:
        # tile images
        tiled_images = tiler.tile_image(images, batched=True)
        tiled_labels = tiler.tile_image(labels, batched=True)

        # cut out parts of the input image
        mask = torch.ones_like(tiled_images)
        for i in range(tiled_images.size(0)):
            h = torch.randint(1, max_h, (1,)).item()
            w = torch.randint(1, max_w, (1,)).item()
            top = torch.randint(0, 224 - h, (1,)).item()
            left = torch.randint(0, 224 - w, (1,)).item()
            mask[i, :, top:top+h, left:left+w] = 0

        masked_images = tiled_images * mask

        # prepare for model input
        if args.debug:
            model_input = preprocess(masked_images)[0, :, :, :].to(device).unsqueeze(0)
            labels = tiled_labels[0, :, :, :].to(device).unsqueeze(0)
        else:
            model_input = preprocess(masked_images).to(device)
            labels = tiled_labels.to(device)

        # learning step
        output = model(model_input)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        # record loss
        running_loss += loss.cpu().item()

    # record loss
    running_loss /= len(dataloader)
    epoch_losses[epoch] = running_loss
    print(f"Loss: {running_loss}\n")

    # evaluate on the test image and save output every epoch
    model.eval()
    tiled_image = tiler.tile_image(test_img, batched=True)
    masked_image = tiled_image * test_mask
    model_input = preprocess(masked_image).to(device)
    with torch.no_grad():
        output = model(model_input)

    output = tiler.stitch_tiles(output.detach().cpu(), batched=True)
    output = inv_normalize(output).squeeze().detach().cpu().permute(1, 2, 0).numpy()

    masked_image = tiler.stitch_tiles(masked_image.cpu(), batched=True)
    masked_image = inv_normalize(masked_image).squeeze().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(masked_image)
    axs[1].imshow(output)
    axs[0].axis("off")
    axs[1].axis("off")
    plt.savefig(f"{args.output}test_epoch_{epoch}.png")
    plt.close()

# make dataframe of losses and save
training_metrics = pl.DataFrame({
    "epoch_loss": epoch_losses
    })
training_metrics.write_csv(f"{args.output}training_metrics.csv")
