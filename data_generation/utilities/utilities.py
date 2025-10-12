"""
Utility functions to use for synthetic data generation
"""

import warnings
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

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
        print(f"[INFO] Resized image to (2048, 3072) and saved to {img_path}")

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
    
