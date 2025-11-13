"""Builds a median-based background model and uses this for background subtraction"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utilities.image_utils import write_to_video
from tqdm import tqdm

# get frame names
dir = "../real_data/fixed_window_frames/"
frame_names = sorted(os.listdir(dir))
paths = [dir + frame_name for frame_name in frame_names]

# loop through, subtract, and make video
frames = []
prev_frame = Image.open(paths[0]).convert('L')
prev_frame = np.array(prev_frame)
background_model = prev_frame
for i, path in tqdm(enumerate(paths)):
    if i == 0:
        continue
    curr_frame = Image.open(path).convert('L')
    curr_frame = np.array(curr_frame)

