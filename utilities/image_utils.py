"""
Utility functions to use for synthetic data generation
"""

import warnings
import numpy as np
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

def write_to_video(frames, file_path, color=True):
    out = cv2.VideoWriter(file_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          30, 
                          (frames[0].shape[1], frames[0].shape[0]), 
                          isColor=color)

    # frames = [frame.astype(np.uint8) * 255 for frame in frames]
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {file_path}!")

# Load Image functions
def load_and_resize_image(img_path, target_size=(3072, 2048)):
    """For images taken with a different camera. Resizes to match resolution of our high-res camera"""
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            warnings.warn("Loaded image is not RGB! Is this intentional?")
        resized = ImageOps.fit(
                img,
                target_size,
                method=Image.Resampling.BICUBIC
                )
        resized.save(img_path)
        print(f"[INFO] Resized image from ({img.height}, {img.width}) to (2048, 3072) and saved to {img_path}")

    return np.array(resized)

def load_image(img_path):
    """For images taken with our 6mp camera that we will use for the robot"""
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            warnings.warn("Loaded image is not RGB! Is this intentional?")
        return np.array(img)

# Show image functions
def show_image(image, title="Image"):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    plt.show()

def compare_images(images, titles=[]):
    """
    Plot multiple images side by side

    images: a list of numpy arrays 
    titles: list of titles (strings) for each image
    """
    fig, axs = plt.subplots(1, len(images), figsize=(9*len(images),6))
    for i in range(len(images)):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        axs[i].axis("off")
    plt.show()
    
# Block images
class BlockCoords:
    def __init__(self, row_start, row_stop, col_start, col_stop):
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop

    def get_center(self, max_row):
        return (self.col_start + self.col_stop) / 2, max_row - (self.row_start + self.row_stop) / 2

    def shift(self, row_shift, col_shift):
        row_start = self.row_start + row_shift
        row_stop = self.row_stop + row_shift
        col_start = self.col_start + col_shift
        col_stop = self.col_stop + col_shift

        shifted = BlockCoords(row_start, row_stop, col_start, col_stop)

        return shifted

    def __sub__(self, other):
        if isinstance(other, BlockCoords):
            return (self.row_start - other.row_start, self.col_start - other.col_start)
        else:
            raise TypeError("not supported")

    def __str__(self):
        return f"{self.row_start} {self.row_stop}\n{self.col_start} {self.col_stop}"

def get_blocks(image, block_size):
    """returns a list of blocks in row-major format"""
    H, W, C = image.shape
    n_h_tiles = int(H / block_size)
    n_w_tiles = int(W / block_size)
    coords = []
    tiles = []
    for row in range(n_h_tiles):
        for col in range(n_w_tiles):
            row_start = row * block_size
            row_stop = (row+1) * block_size
            col_start = col * block_size
            col_stop = (col+1) * block_size
            coords.append(BlockCoords(row_start, row_stop, col_start, col_stop))
            tiles.append(image[row_start:row_stop, col_start:col_stop, :])
    return tiles, coords

# Visualize fourier transform
def ft_viz(ft):
    return np.log(np.abs(ft) + 1)

