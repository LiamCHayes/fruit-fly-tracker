"""Resize all frames to the 6mp size of the camera"""

import os
import argparse
from utilities import load_and_resize_image

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of the frames directory to resize")
args = parser.parse_args()

# file path
file_names = os.listdir(args.name)
file_paths = [args.name + name for name in file_names]

# resize images
for path in file_paths:
    load_and_resize_image(path)

