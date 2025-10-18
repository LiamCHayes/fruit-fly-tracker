"""
Generates labelled fruit fly scenes with black dots as fruit flies

1000 images from each background
"""

import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from utilities.image_utils import load_and_resize_image, load_image, show_image, compare_images

# Draw samples for fruit fly location
def sample_fly_positions(n_samples, fly_radius):
    """Generate random data to place flies"""
    num_flies = np.random.randint(0, 10, size=n_samples)

    fly_positions = [None for _ in range(n_samples)]
    for i in range(n_samples):
        row_pos = np.random.randint(fly_radius, 2048-fly_radius, size=num_flies[i])
        col_pos = np.random.randint(fly_radius, 3072-fly_radius, size=num_flies[i])
        fly_position_list = np.stack((row_pos, col_pos), axis=1)
        fly_positions[i] = np.round(fly_position_list).astype(int)

    return fly_positions

# Place fruit flies
def draw_flies(background, fly_positions, fly_radius):
    """Draws flies on the background"""
    new_image = background.copy()
    row, col, _ = new_image.shape
    mask = np.zeros((row, col))
    for row, col in fly_positions:
        fly_rows = np.arange(row - fly_radius, row + fly_radius)
        fly_cols = np.arange(col - fly_radius, col + fly_radius)
        new_image[np.ix_(fly_rows, fly_cols)] = [0, 0, 0]

        # Generate segmentation mask
        mask[np.ix_(fly_rows, fly_cols)] = 1

    return new_image, mask

if __name__ == "__main__":
    # Choose things for the generated data 
    background_labels = os.listdir("dataset/backgrounds")
    n_datapoints = 1000
    fly_radius = 5
    train = True # if true, will save to the train dataset folder
    if train:
        dataset_folder = "train"
    else:
        dataset_folder = "test"

    for label in background_labels:
        # Load and check if the image is the correct size
        print(f"Generating flies on {label}...")
        img_path = f"dataset/backgrounds/{label}"
        background = load_image(img_path)
        if background.shape != (2048, 3072, 3):
            background = load_and_resize_image(img_path)

        # Generate random fly positions
        fly_positions = sample_fly_positions(n_datapoints, fly_radius)

        for i in tqdm(range(n_datapoints)):
            synthetic_img, mask = draw_flies(background, fly_positions[i], fly_radius)
            # compare_images([synthetic_img, mask], ["Synthetic image", "mask"])

            # Save to dataset
            datapoint_name = f"{label.split(".")[0]}-{str(i).zfill(5)}"

            image = Image.fromarray(synthetic_img)
            image.save(f"dataset/{dataset_folder}/images/{datapoint_name}.jpeg")

            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask.save(f"dataset/{dataset_folder}/masks/{datapoint_name}.png")

