"""Cut out flies from an image and make a sticker to paste the fly over other backgrounds, generating synthetic images"""

from operator import pos
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

# Get frames to loop through to make fly cutouts
path_to_dir = "../real_data/fruit_fly_cabinet_frames/"
frame_name = os.listdir(path_to_dir)
paths = [path_to_dir + name for name in frame_name]

# Loop through frames
for i, path in enumerate(paths):
    # load image
    fly_image = Image.open(path)
    fly_image_numpy = np.array(fly_image)

    # get bounding box of fly
    fig, ax = plt.subplots(constrained_layout = True, figsize=(12, 10))
    ax.imshow(fly_image_numpy)
    klicker = clicker(ax, ["fly"], markers=["o"])
    plt.show()
    positions = klicker.get_positions()

    print(positions)
    accept = input("Accept?")
    if accept == "":
        fly = positions['fly']
        left = int(fly[0, 0])
        upper = int(fly[0, 1])
        right = int(fly[1, 0])
        lower = int(fly[1, 1])
        print("Accepted")
    else:
        print("Rejected")
        continue

    # crop fly
    box = (left, upper, right, lower)
    cropped_fly = fly_image.crop(box)

    # save fly sticker to fly bank
    cropped_fly.save(f"dataset/flies/{frame_name[i]}.png", format="PNG")

