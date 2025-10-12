"""
Generates labelled fruit fly scenes with black dots as fruit flies

1000 images from each background
"""

import numpy as np
from PIL import Image, ImageOps
from utilities import load_and_resize_image, load_image, show_image, compare_images

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
    background_label = "liam_veggies"
    img_path = f"dataset/backgrounds/{background_label}.jpg"
    n_datapoints = 1000
    fly_radius = 5

    # Load and check if the image is the correct size
    background = load_image(img_path)
    if background.shape != (2048, 3072, 3):
        background = load_and_resize_image(img_path)
    # show_image(background, "Original background")

    fly_positions = sample_fly_positions(n_datapoints, fly_radius)

    for i in range(n_datapoints):
        synthetic_img, mask = draw_flies(background, fly_positions[i], fly_radius)
        # compare_images([synthetic_img, mask], ["Synthetic image", "mask"])
        
        # Save to dataset
        datapoint_name = f"{background_label}-{str(i).zfill(5)}"

        image = Image.fromarray(synthetic_img)
        image.save(f"dataset/images/{datapoint_name}.jpg")

        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        mask.save(f"dataset/masks/{datapoint_name}.png")

