"""Animate the frequency space of a moving fruit fly video"""

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from utilities.filters import high_pass

# load grapes frames
dir_path = "../real_data/grapes_frames/"
paths = [dir_path + path for path in os.listdir(dir_path)]
paths.sort()

# video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("fft_plots/grapes_video_filtered.avi", fourcc, 30, (3072, 1024))

def ft_viz(ft):
    return np.log(np.abs(ft) + 1)

frames = []
for path in tqdm(paths):
    # load image
    frame = Image.open(path)
    frame = np.array(frame)
    gray_frame = np.mean(frame, axis=2)

    # get corresponding global fourier transform
    ft = np.fft.fft2(gray_frame)
    ft_shifted = np.fft.fftshift(ft)
    ft_filtered = high_pass(ft_shifted)

    # make frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    ft_viz_frame = ft_viz(ft_filtered)
    ft_viz_norm = cv2.normalize(ft_viz_frame, None, 0, 255, cv2.NORM_MINMAX)
    ft_viz_uint8 = ft_viz_norm.astype(np.uint8)
    ft_3ch = cv2.cvtColor(ft_viz_uint8, cv2.COLOR_GRAY2BGR)

    full_frame = np.hstack((frame_bgr, ft_3ch))

    # downsample
    scale_percent = 50  # percent of original size (e.g., 50%)
    width = int(full_frame.shape[1] * scale_percent / 100)
    height = int(full_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    downsampled_frame = cv2.resize(full_frame, dim, interpolation=cv2.INTER_AREA)

    # animate
    video.write(downsampled_frame)

video.release()
