"""Frame differencing algorithm for two frames"""

import os
import copy
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

def get_intensity(frame):
    print(frame)
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

def dilation(frames, kernel_size, iters):
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dil_list = []
    for frame in frames:
        dilated_frame = cv2.dilate(frame, dilation_kernel, iterations=iters)
        dil_list.append(dilated_frame)

    return np.stack(dil_list)

def erosion(frames, kernel_size, iters):
    erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ero_list = []
    for frame in frames:
        eroded_frame = cv2.erode(frame, erosion_kernel, iterations=iters)
        ero_list.append(eroded_frame)

    return np.stack(ero_list)


def threshold_test(frames, variance_s, mrf_order, T, n_iter, ero_dil=False):
    """
    Implement fixed threshold hypothesis test
    Takes a list of frames (numpy arrays) and returns a list of realizations based on the fixed threshold hypothesis test
    """
    # Create arrays of frames
    masks = []
    for frame in frames:
        fore = get_intensity(frame[0])
        back = get_intensity(frame[1])

        # Subtract frames
        differences = np.sqrt((fore - back) **2)

        # Calculate threshold
        theta = 1 # assumed from instructions
        L = 255 # luminance range
        threshold = 2 * variance_s * np.log(2 * L * theta / np.sqrt(2 * np.pi * variance_s))

        mask = differences > threshold

        if ero_dil == True:
            mask = (mask * 1).astype(np.uint8)
            #opening
            mask = erosion(mask, 2, 1)
            mask = dilation(mask, 2, 1)

            #closing
            mask = dilation(mask, 2, 1)
            mask = erosion(mask, 2, 1)

        if mrf_order == 1:
            mask = first_order_mrf(differences, mask, variance_s, T, n_iter)
        elif mrf_order == 2:
            mask = second_order_mrf(differences, mask, variance_s, T, n_iter)

        masks.append(mask)
    return masks

def load_src_images(filename):
    img_list = []
    filenames = os.listdir(f'{filename}/orig/')
    sorted_files = sorted(filenames)
    print(len(sorted_files))
    for file in sorted_files:
        img_list.append(Image.open(f'{filename}/orig/' + file))
    return img_list

def load_mask_images(filename):
    img_list = []
    filenames = os.listdir(f'{filename}/background/')
    sorted_files = sorted(filenames)
    print(len(sorted_files))
    for file in sorted_files:
        img_list.append(Image.open(f'{filename}/background/' + file))
    return img_list


if __name__ == "__main__":
    # Create list of frames
    filename = "grapes"
    sorted_src_images = load_src_images(filename)
    sorted_mask_images = load_mask_images(filename)
    sorted_images = (sorted_src_images, sorted_mask_images)
    img_list = []
    
    for i, (src_image, mask_image) in enumerate(zip(sorted_src_images, sorted_mask_images)):
        img_list.append([src_image, mask_image, i])

    # Hyperparameters
    variance_s = 3
    mrf_order = 1
    T = 1
    n_iter = 2

    # fixed threshold test
    print("fixed threshold...")
    fixed_mask_frames = threshold_test(img_list, variance_s, 0, 0, 0)
    write_to_video(fixed_mask_frames, f"{filename}_fixed_threshold.mp4", False)

    # fixed threshold test w/ erosion and dilation
    print("erosion and dilation...")
    fixed_mask_frames= threshold_test(img_list, variance_s, 0, 0, 0, True)
    write_to_video(fixed_mask_frames, f"{filename}_erosion_dilation.mp4", False)

