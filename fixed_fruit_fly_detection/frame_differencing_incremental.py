"""Incrementally does frame differencing. Less memory intesive"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utilities.image_utils import write_to_video, show_image
from tqdm import tqdm

# get frame names
dir = "../real_data/fixed_window_frames/"
frame_names = sorted(os.listdir(dir))
paths = [dir + frame_name for frame_name in frame_names]

# loop through, subtract, and make video
frames = []
prev_frame = Image.open(paths[0]).convert('L')
prev_frame = np.array(prev_frame).astype(np.float32)
for i, path in tqdm(enumerate(paths)):
    if i == 0:
        continue
    curr_frame = Image.open(path).convert('L')
    curr_frame = np.array(curr_frame).astype(np.float32)

    subtracted = (curr_frame - prev_frame)**2
    thresholded = subtracted > 90
    thresholded = thresholded.astype(np.uint8) * 255

    # Combine the frames
    top_row = np.concatenate([prev_frame, curr_frame], axis=1)
    bottom_row = np.concatenate([subtracted, thresholded], axis=1)
    full_frame = np.concatenate([top_row, bottom_row], axis=0).astype(np.uint8)

    # downsample to make a reasonable sized video
    full_frame = Image.fromarray(full_frame, mode='L')
    width, height = full_frame.size
    downsampled = full_frame.resize((width // 2, height // 2))
    frames.append(np.array(downsampled))

    prev_frame = curr_frame

write_to_video(frames, "output/frame_differencing.mp4", False)
