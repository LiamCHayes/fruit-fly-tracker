"""Fine tune the self-supervised decoder on synthetic data"""

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
from synthetic_dataset import SyntheticData
from tile_picture import ImageTiler

# required arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", help="path to the directory of images (end with /)", required=True)
parser.add_argument("-m", "--masks", help="path to the directory of masks (end with /)", required=True)
parser.add_argument("-o", "--output", help="path to the output directory (end with /)", required=True)
parser.add_argument("-w", "--weights", help="path to the pre-trained encoder weights", required=True)
parser.add_argument("-t", "--test", help="path to the test image", required=True)

# default arguments
parser.add_argument("--debug", action="store_true", help="make model input 1 tile for fast forward passes")
parser.add_argument("-e", "--epochs", default=256, help="number of epochs")
parser.add_argument("-b", "--batch-size", default=1, help="batch size")
parser.add_argument("-c", "--checkpoint-interval", default=1, help="When to save model checkpoints")
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

# Hyperparameters
epochs = args.epochs
batch_size = args.batch_size
max_h = 128
max_w = 128
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# model training things
encoder = ResNet50Encoder()
encoder_state_dict = torch.load(args.weights)
encoder.load_state_dict(encoder_state_dict)
model = ResNet50Decoder(1, encoder)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr= 0.0001)

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice
        return bce + dice_loss
loss_fn = DiceBCELoss()

# load the dataset and image tiler
dataset = SyntheticData(args.images, args.masks)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
tiler = ImageTiler()

# load the test image
with Image.open(args.test) as img:
    test_img = dataset.img_transform(img).unsqueeze(0)

# training loop
epoch_losses = [0.0 for _ in range(epochs)]
min_loss = None
for epoch in range(epochs):
    model.train()
    running_loss = 0
    print(f"Epoch {epoch}")

    for images, labels in dataloader:
        # tile images
        tiled_images = tiler.tile_image(images, batched=True)
        tiled_labels = tiler.tile_image(labels, batched=True)

        # prepare for model input
        if args.debug:
            # only input one tile for faster forward pass
            model_inputs = [preprocess(tiled_images)[0, :, :, :].unsqueeze(0)]
            split_labels = [tiled_labels[0, :, :, :].unsqueeze(0)]
        else:
            # split the tiles into two batches so we can run on 16GB gpu
            model_inputs = torch.split(preprocess(tiled_images), 70, dim=0)
            split_labels = torch.split(tiled_labels, 70, dim=0)

        # learning step(s)
        for i in range(len(model_inputs)):
            model_input = model_inputs[i].to(device)
            label = split_labels[i].to(device)

            output = model(model_input)
            optimizer.zero_grad()
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            # record loss
            running_loss += loss.cpu().item()

    # record loss
    running_loss /= len(dataloader)
    epoch_losses[epoch] = running_loss
    print(f"Loss: {running_loss}\n")

    if min_loss is None:
        min_loss = running_loss

    # Every few epochs
    if (epoch+1) % args.checkpoint_interval == 0 or min_loss > running_loss:
        # generate file paths for saving
        if min_loss > running_loss:
            min_loss = running_loss
            full_model_path = f"{args.output}full_model_best_loss.pth"
            encoder_path = f"{args.output}encoder_best_loss.pth"
            viz_path = f"{args.output}test_img_best_loss.png"
        else:
            full_model_path = f"{args.output}full_model_epoch_{epoch}.pth"
            encoder_path = f"{args.output}encoder_epoch_{epoch}.pth"
            viz_path = f"{args.output}test_epoch_{epoch}.png"

        # Save checkpoint model
        model.save_checkpoint(full_model_path, encoder_path)

        # evaluate on the test image and save output
        model.eval()
        tiled_image = tiler.tile_image(test_img, batched=True)
        model_input = preprocess(tiled_image).to(device)
        with torch.no_grad():
            output = model(model_input)

        output = tiler.stitch_tiles(output.detach().cpu(), batched=True)
        output = inv_normalize(output).squeeze().detach().cpu().permute(1, 2, 0).numpy()
        orig_img = test_img.squeeze().cpu().permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(orig_img)
        axs[1].imshow(output)
        axs[0].axis("off")
        axs[1].axis("off")
        plt.savefig(viz_path)
        plt.close()

        # make dataframe of losses and save
        training_metrics = pl.DataFrame({
            "epoch_loss": epoch_losses
            })
        training_metrics.write_csv(f"{args.output}training_metrics.csv")

# make dataframe of losses and save
training_metrics = pl.DataFrame({
    "epoch_loss": epoch_losses
    })
training_metrics.write_csv(f"{args.output}training_metrics.csv")
