"""
Generates labeled fruit fly scenes with black dots as fruit flies

1000 images from each background
"""

import os
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

# Place fruit flies
def add_fly(image, row, col, axes, angle_deg, color=(20,20,20), blur_sigma=1.5):
    H, W = image.shape[:2]

    # Create a grid in the region around the fly
    y, x = np.ogrid[:H, :W]
    angle = np.deg2rad(angle_deg)

    # Ellipse formula (rotated)
    x_rot = (x - col) * np.cos(angle) + (y - row) * np.sin(angle)
    y_rot = -(x - col) * np.sin(angle) + (y - row) * np.cos(angle)
    ellipse_mask = ((x_rot/axes[0])**2 + (y_rot/axes[1])**2) <= 1

    # Darken pixels in the ellipse region
    for c in range(3):
        image[..., c][ellipse_mask] = color[c]
    padding = int(3 * blur_sigma)
    min_row = max(0, row - axes[1] - padding)
    max_row = min(H, row + axes[1] + padding)
    min_col = max(0, col - axes[0] - padding)
    max_col = min(W, col + axes[0] + padding)

    sub_img = image[min_row:max_row, min_col:max_col].copy()
    sub_mask = ellipse_mask[min_row:max_row, min_col:max_col]

    # Apply Gaussian blur only on the sub-image
    for c in range(3):
        channel = sub_img[..., c]
        blurred_channel = gaussian_filter(channel, sigma=blur_sigma)

        # Blend blurred and original based on mask - blur edges only
        channel[sub_mask] = blurred_channel[sub_mask]
        sub_img[..., c] = channel

    # Put blurred region back in the original image
    image[min_row:max_row, min_col:max_col] = sub_img

    return image, ellipse_mask

def draw_flies(background, fly_positions):
    """Draws flies on the background"""
    new_image = background.copy()
    row, col, _ = new_image.shape
    bbox_labels = []
    for row, col in fly_positions:
        axes = (np.random.randint(6, 12), np.random.randint(4, 8))
        angle = np.random.randint(0, 180)
        new_image, mask_addition = add_fly(new_image, row, col, axes, angle)

        # get yolo bounding box label
        fly_pixels = np.where(mask_addition==True)
        fly_rows = fly_pixels[0]
        fly_cols = fly_pixels[1]
        fly_width = np.max(fly_cols) - np.min(fly_cols)
        fly_height = np.max(fly_rows) - np.min(fly_rows)
        bbox_labels.append([col, row, fly_width, fly_height])

    return new_image, bbox_labels

if __name__ == "__main__":
    # Choose things for the generated data 
    background_labels = os.listdir("dataset/backgrounds")
    n_datapoints = 50
    fly_radius = 5
    dataset_folder = "yolo_dataset" # Folder in the dataset directory to save the images and masks to

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
            synthetic_img, bbox_labels = draw_flies(background, fly_positions[i])
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

                image = Image.fromarray(synthetic_img)
                image.save(f"dataset/{dataset_folder}/images/{set_name}/{datapoint_name}.jpeg")

                with open(f"dataset/{dataset_folder}/labels/{set_name}/{datapoint_name}.txt", "w") as file:
                    for f in range(flies_in_tile.shape[0]):
                        file.write(f"0 {flies_in_tile[f, 0]} {flies_in_tile[f, 1]} {flies_in_tile[f, 2]} {flies_in_tile[f, 3]}\n")

