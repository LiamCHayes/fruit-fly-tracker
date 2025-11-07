"""Frame differencing algorithm for two frames"""

import os
import copy
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

def get_intensity(frame):
    intensity = np.mean(frame, axis=2)
    return intensity

def write_to_video(frames, file_path, color=True):
    out = cv2.VideoWriter(file_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          30, 
                          (frames[0].shape[1], frames[0].shape[0]), 
                          isColor=color)

    frames = [frame.astype(np.uint8) * 255 for frame in frames]
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {file_path}!")

def subtract_background(frames, N):
    """Computes background difference"""
    # Create arrays of frames
    intensities = [get_intensity(frame) for frame in frames]
    backgrounds = []

    # get the median background
    for i in range(len(intensities)):
        start_idx = np.max(0, i-N)
        median_frames = intensities[start_idx:i]
        median_array = np.stack(median_frames, axis=-1)
        median_frame = np.median(median_array, axis=-1)
        backgrounds.append(median_frame)

    intensities = np.stack(intensities, axis=-1)
    backgrounds = np.stack(backgrounds, axis=-1)

    differences = (intensities - backgrounds) ** 2
    return differences

def first_order_mrf(q_k, mask, variance_s, T, n_iter):
    term_1 = 2 * variance_s
    term_2 = np.log(5)
    for _ in range(n_iter):
        # Calculate threshold
        threshold = copy.deepcopy(q_k).astype(np.float64)
        for i in tqdm(range(1, mask.shape[0]-1)):
            for j in range(1, mask.shape[1]-1):
                neighborhood_idxs = [[i-1, j],
                                     [i+1, j],
                                     [i, j-1],
                                     [i, j+1]]
                for t in range(mask.shape[2]):
                    mask_frame = mask[:, :, t]
                    e_k = [mask_frame[idx[0], idx[1]] for idx in neighborhood_idxs]
                    R_m = np.sum(e_k)
                    R_s = len(e_k) - R_m
                    threshold[i, j, t] = term_1 * (term_2 + (R_s - R_m) / T)
        mask = q_k > threshold

    return mask

def second_order_mrf(q_k, mask, variance_s, T, n_iter):
    term_1 = 2 * variance_s
    term_2 = np.log(5)
    for _ in range(n_iter):
        # Calculate threshold
        threshold = copy.deepcopy(q_k).astype(np.float64)
        for i in tqdm(range(1, mask.shape[0]-1)):
            for j in range(1, mask.shape[1]-1):
                neighborhood_idxs = [[i-1, j],
                                     [i+1, j],
                                     [i, j-1],
                                     [i, j+1],
                                     [i-1, j-1],
                                     [i+1, j-1],
                                     [i-1, j+1],
                                     [i+1, j+1]]
                for t in range(mask.shape[2]):
                    mask_frame = mask[:, :, t]
                    e_k = [mask_frame[idx[0], idx[1]] for idx in neighborhood_idxs]
                    R_m = np.sum(e_k)
                    R_s = len(e_k) - R_m
                    threshold[i, j, t] = term_1 * (term_2 + (R_s - R_m) / T)
        mask = q_k > threshold

    return mask

def threshold_test(frames, variance_s, mrf_order, T, n_iter):
    """
    Implement fixed threshold hypothesis test
    Takes a list of frames (numpy arrays) and returns a list of realizations based on the fixed threshold hypothesis test
    """
    # Create arrays of frames
    intensities = [get_intensity(frame) for frame in frames]
    prev_intensities = copy.deepcopy(intensities)
    prev_intensities.pop(-1)
    intensities.pop(0)
    intensities = np.stack(intensities, axis=-1)
    prev_intensities = np.stack(prev_intensities, axis=-1)

    # Subtract frames
    differences = (intensities - prev_intensities) ** 2

    # Calculate threshold
    theta = 1 # assumed from instructions
    L = 255 # luminance range
    threshold = 2 * variance_s * np.log(2 * L * theta / np.sqrt(2 * np.pi * variance_s))

    mask = differences > threshold

    if mrf_order == 1:
        mask = first_order_mrf(differences, mask, variance_s, T, n_iter)
    elif mrf_order == 2:
        mask = second_order_mrf(differences, mask, variance_s, T, n_iter)

    mask_frames = np.split(mask, mask.shape[2], axis=2)
    return mask_frames, np.split(differences, differences.shape[2], axis=2)

if __name__ == "__main__":
    # Create list of frames
    video_name = "fruit_fly_vial"
    input_dir = f"../real_data/{video_name}_frames/"
    file_paths = [input_dir+fn for fn in sorted(os.listdir(input_dir))]
    frames = [np.array(Image.open(fp)) for fp in file_paths]

    # Hyperparameters
    variance_s = 10
    mrf_order = 1
    T = 1
    n_iter = 2

    # fixed threshold test
    print("fixed threshold...")
    fixed_mask_frames, differences = threshold_test(frames, variance_s, 0, 0, 0)
    write_to_video(fixed_mask_frames, f"output/{video_name}_fixed_threshold.mp4", False)

    # first order mrf test
    print("first order mrf...")
    first_mask_frames, _ = threshold_test(frames, variance_s, 1, T, n_iter)
    write_to_video(first_mask_frames, f"output/{video_name}_first_order.mp4", False)

    # second order mrf test
    print("second order mrf...")
    second_mask_frames, _ = threshold_test(frames, variance_s, 2, T, n_iter)
    write_to_video(second_mask_frames, f"output/{video_name}_second_order.mp4", False)

