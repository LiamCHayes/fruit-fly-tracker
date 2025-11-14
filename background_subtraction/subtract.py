"""Subtract the current frame from a background model"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utilities.image_utils import write_to_video, show_image
from utilities.filters import high_pass, low_pass
from tqdm import tqdm

# get background frame names
dir = "../real_data/background_estimation_desk/"
frame_names = sorted(os.listdir(dir))
background_paths = [dir + frame_name for frame_name in frame_names]
background_paths.pop(-1)

# get actual frame names
dir = "../real_data/fruit_fly_desk_frames/"
frame_names = sorted(os.listdir(dir))
actual_paths = [dir + frame_name for frame_name in frame_names]
actual_paths = actual_paths[:101]

# loop through, subtract, and make video
frames = []
for i in tqdm(range(len(background_paths))):
    background = Image.open(background_paths[i]).convert('L')
    background = np.array(background).astype(np.float32)

    actual = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR).convert('L')
    actual = np.array(actual).astype(np.float32)

    # Spatial domain subtraction and thresholding
    spatial_subtracted = (actual - background)**2

    spatial_thresholded = spatial_subtracted > 400
    spatial_thresholded = spatial_thresholded.astype(np.uint8) * 255

    # Get frequency domain
    background_ft = np.fft.fft2(background)
    actual_ft = np.fft.fft2(actual)

    # Subtract and threshold unfiltered frequencies
    ft_subtracted = actual_ft - background_ft
    ft_img_subtracted = np.abs(np.fft.ifft2(ft_subtracted))

    ft_img_thresholded = ft_img_subtracted > 25
    ft_img_thresholded = ft_img_thresholded.astype(np.uint8) * 255

    # Choose which frames to add to the video
    top_left = background
    top_right = actual
    bottom_left = spatial_thresholded
    bottom_right = ft_img_thresholded

    # Combine the frames
    top_row = np.concatenate([top_left, top_right], axis=1)
    bottom_row = np.concatenate([bottom_left, bottom_right], axis=1)
    full_frame = np.concatenate([top_row, bottom_row], axis=0).astype(np.uint8)

    # downsample to make a reasonable sized video
    full_frame = Image.fromarray(full_frame, mode='L')
    width, height = full_frame.size
    downsampled = full_frame.resize((width // 2, height // 2))
    frames.append(np.array(downsampled))

write_to_video(frames, "output/frame_differencing.mp4", False)
