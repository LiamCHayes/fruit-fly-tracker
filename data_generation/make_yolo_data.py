"""
Generates labeled fruit fly scenes with black dots as fruit flies

1000 images from each background
"""

import os
import random
import numpy as np
from PIL import Image
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from utilities.image_utils import load_and_resize_image, load_image, show_image, compare_images, get_blocks

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

# Blends the fly into the background with an alpha mask
def create_alpha_mask(image, threshold=160):
    # Split image into channels
    r, g, b, a = image.split()
    # Create a new alpha mask
    new_alpha = Image.new("L", image.size, 0)

    # Load pixel data
    pixels = image.load()
    alpha_pixels = new_alpha.load()

    # Loop through pixels and set alpha channel
    for y in range(image.height):
        for x in range(image.width):
            r_val, g_val, b_val, _ = pixels[x, y]
            # Calculate brightness
            brightness = (r_val + g_val + b_val) // 3
            if brightness < threshold:  # Threshold for dark pixels
                alpha_pixels[x, y] = 255  # opaque
            else:
                alpha_pixels[x, y] = 0    # transparent

    return new_alpha

# Place a single fruit fly
def add_fly(background, cropped_fly, row, col, angle):
    # create alpha mask
    cropped_fly = cropped_fly.convert("RGBA")
    alpha_mask = create_alpha_mask(cropped_fly)

    # Paste the fly onto the background at position (x,y)
    # TODO rotate the fly by angle?
    position = (col, row)
    background.paste(cropped_fly, position, mask=alpha_mask)

    # get height, width of the cropped fly and get bounding box
    width, height = cropped_fly.size
    bbox_coords = [col, row, width, height]

    return background, bbox_coords

def draw_flies(background, fly_positions, fly_dir):
    """Draws flies on the background"""
    new_image = background.copy()
    row, col, _ = new_image.shape
    bbox_labels = []
    for row, col in fly_positions:
        # sample a fly from the cropped flies
        cropped_fly_path = random.choice(fly_dir)
        cropped_fly = Image.open(cropped_fly_path)

        # paste fly
        angle = np.random.randint(0, 180)
        new_image, bbox_coords = add_fly(new_image, cropped_fly, row, col, angle)

        # get yolo bounding box label
        bbox_labels.append(bbox_coords)

    return new_image, bbox_labels

if __name__ == "__main__":
    # Choose things for the generated data 
    background_labels = os.listdir("dataset/backgrounds")
    n_datapoints = 50
    fly_radius = 5
    dataset_folder = "yolo_dataset" # Folder in the dataset directory to save the images and masks to

    cropped_flies = os.listdir("dataset/flies")
    cropped_flies = ["dataset/flies" + fly for fly in cropped_flies]

    # Loop through the backgrounds included for this dataset
    for label in background_labels:
        # Load and check if the image is the correct size
        print(f"Generating flies on {label}...")
        img_path = f"dataset/backgrounds/{label}"
        background = load_image(img_path)
        if background.shape != (2048, 3072, 3):
            background = load_and_resize_image(img_path) 

        set_name = "val" if label == "counter_with_pen.jpeg" else "train"

        # Generate random fly positions
        fly_positions = sample_fly_positions(n_datapoints, fly_radius)

        # Loop through randomly generated fly positions
        for i in tqdm(range(n_datapoints)):
            synthetic_img, bbox_labels = draw_flies(background, fly_positions[i], cropped_flies)
            # compare_images([synthetic_img, mask], ["Synthetic image", "mask"])

            # tile images for input to yolo
            tiles, coords = get_blocks(synthetic_img, 640)

            normalized_labels = []
            for j in range(len(tiles)):
                # edit bounding box labels to correspond to new tile coordinates
                flies_in_tile = []
                tile_coordinates = coords[j]
                for fly in bbox_labels:
                    in_x = fly[0] in range(tile_coordinates.col_start, tile_coordinates.col_stop)
                    in_y = fly[1] in range(tile_coordinates.row_start, tile_coordinates.row_stop)
                    if in_x and in_y:
                        flies_in_tile.append(fly)
                flies_in_tile = np.array(flies_in_tile)
                if len(flies_in_tile) > 0:
                    flies_in_tile[:,0] = flies_in_tile[:,0] - tile_coordinates.col_start
                    flies_in_tile[:,1] = flies_in_tile[:,1] - tile_coordinates.row_start
                    flies_in_tile = flies_in_tile / 640

                # TODO apply data augmentations

                # Save tile and label
                datapoint_name = f"{label.split(".")[0]}-{str(i).zfill(5)}-{str(j).zfill(2)}"

                image = Image.fromarray(tiles[j])
                image.save(f"dataset/{dataset_folder}/images/{set_name}/{datapoint_name}.jpeg")

                with open(f"dataset/{dataset_folder}/labels/{set_name}/{datapoint_name}.txt", "w") as file:
                    for f in range(flies_in_tile.shape[0]):
                        file.write(f"0 {flies_in_tile[f, 0]} {flies_in_tile[f, 1]} {flies_in_tile[f, 2]} {flies_in_tile[f, 3]}\n")

