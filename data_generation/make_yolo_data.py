"""
Generates labeled fruit fly scenes with black dots as fruit flies

1000 images from each background
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
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
def add_fly(background, cropped_fly, row, col):
    # create alpha mask
    cropped_fly = cropped_fly.convert("RGBA")
    alpha_mask = create_alpha_mask(cropped_fly)
    alpha_mask = np.array(alpha_mask)
    mask = cv2.threshold(alpha_mask, 1, 255, cv2.THRESH_BINARY)[1]
    width, height = cropped_fly.size

    # increase contrast of the fly
    cropped_fly = cropped_fly.convert("RGB")
    enhancer = ImageEnhance.Contrast(cropped_fly)
    cropped_fly = enhancer.enhance(2.5)
    cropped_fly = np.array(cropped_fly)

    # convert color spaces to bgr
    cropped_fly = cv2.cvtColor(cropped_fly, cv2.COLOR_RGB2BGR)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

    # Paste the fly onto the background at position (x,y)
    position = (col, row)
    background2 = cv2.seamlessClone(cropped_fly, background, mask, position, cv2.NORMAL_CLONE)

    # Darken the fly so it's visible
    fly_pixels = np.where(background - background2 != 0)
    background2[fly_pixels[0], fly_pixels[1], :] = background2[fly_pixels[0], fly_pixels[1], :] * 0.1

    # add gaussian blur
    blur_sigma = 2
    dilation_size = 5
    erode_size = 3

    blurred_img = cv2.GaussianBlur(background2, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    fly_mask = np.mean(np.zeros_like(background2), axis=2)
    fly_mask[fly_pixels[0], fly_pixels[1]] = 255.0

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    dilated_mask = cv2.dilate(fly_mask, dilation_kernel, iterations=1)
    eroded_mask = cv2.erode(fly_mask, erode_kernel, iterations=1)

    fly_mask_f = eroded_mask.astype(float) / 255.0
    dilated_mask_f = dilated_mask.astype(float) / 255.0
    transition_mask = dilated_mask_f - fly_mask_f
    transition_mask = np.clip(transition_mask, 0, 1)[:, :, np.newaxis]

    background2 = (
        blurred_img * fly_mask_f[:, :, np.newaxis] +
        (background2 * (1 - transition_mask) + blurred_img * transition_mask) * (1 - fly_mask_f[:, :, np.newaxis])
    ).astype(np.uint8)

    # convert back to rgb
    background2 = cv2.cvtColor(background2, cv2.COLOR_BGR2RGB)

    # return bounding box as well
    bbox_coords = [col, row, width, height]

    return background2, bbox_coords

def draw_flies(background, fly_positions, fly_dir):
    """Draws flies on the background"""
    new_image = background.copy()
    row, col, _ = new_image.shape
    bbox_labels = []
    for row, col in fly_positions:
        # sample a fly from the cropped flies
        cropped_fly_path = random.choice(fly_dir)
        cropped_fly = Image.open(cropped_fly_path)

        new_image, bbox_coords = add_fly(new_image, cropped_fly, row, col)

        # get yolo bounding box label
        bbox_labels.append(bbox_coords)

    return new_image, bbox_labels

if __name__ == "__main__":
    # Choose things for the generated data 
    background_labels = os.listdir("dataset/backgrounds")
    n_datapoints = 50
    fly_radius = 35
    dataset_folder = "yolo_dataset" # Folder in the dataset directory to save the images and masks to

    cropped_flies = os.listdir("dataset/flies")
    cropped_flies = ["dataset/flies/" + fly for fly in cropped_flies]

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

