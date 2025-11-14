"""Visualize frequency space of a fruit fly"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from utilities.filters import high_pass, low_pass
from utilities.image_utils import get_blocks, ft_viz

if __name__ == "__main__":
    fly_seq = "fixed"

    if fly_seq == "grape":
        dir_path = "../real_data/grapes_frames/"
        block_idxs = [69, 72]
    elif fly_seq == "cabinet":
        dir_path = "../real_data/fruit_fly_cabinet_frames/"
        block_idxs = [0, 2, 25, 36, 41, 42, 54, 80, 81, 82]
    elif fly_seq == "fixed":
        dir_path = "../real_data/fixed_window_frames/"
        block_idxs = []
    else:
        print("[ERROR] not a valid frame sequence")
        exit()

    # load imae
    paths = [dir_path + path for path in os.listdir(dir_path)]
    paths.sort()
    path = paths[146]
    frame = Image.open(path)
    frame = np.array(frame)
    gray_frame = np.mean(frame, axis=2)

    # fourier transform
    ft = np.fft.fft2(gray_frame)
    ft_shifted = np.fft.fftshift(ft)
    inverse_img = np.fft.ifft2(ft_shifted)
    inverse_img = np.abs(inverse_img)

    # high pass filter
    ft_filtered = high_pass(ft_shifted, 5)
    ft_ifft_shfited = np.fft.ifftshift(ft_filtered)
    img_filtered = np.fft.ifft2(ft_ifft_shfited)
    img_filtered = np.abs(img_filtered)

    # low pass filter
    ft_low = low_pass(ft_shifted, 10)
    low_img = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_low)))

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs = axs.flatten()
    axs[0].imshow(frame)
    axs[0].set_title("Original block")
    axs[1].imshow(gray_frame, cmap='gray')
    axs[1].set_title("Gray block")
    axs[2].imshow(gray_frame - low_img, cmap='gray')
    axs[2].set_title("Low frequencies subtracted")
    axs[3].imshow(img_filtered, cmap='gray')
    axs[3].set_title("High pass reconstruction")
    plt.tight_layout()
    # plt.savefig(f"cabinet_fly{i}.png")
    plt.show()

    # get blocks
    blocks, block_coords = get_blocks(frame, 256)

    # loop through important blocks
    fts = []
    for i, block in enumerate(blocks):
        # if i not in block_idxs:
           # continue
        print(i)

        # fourier transform of blocks
        gray = np.mean(block, axis=2)
        ft = np.fft.fft2(gray)
        ft_shifted = np.fft.fftshift(ft)
        fts.append(ft_shifted)
        inverse_img = np.fft.ifft2(ft_shifted)
        inverse_img = np.abs(inverse_img)

        # high pass filter
        ft_filtered = high_pass(ft_shifted, 5)
        ft_ifft_shfited = np.fft.ifftshift(ft_filtered)
        img_filtered = np.fft.ifft2(ft_ifft_shfited)
        img_filtered = np.abs(img_filtered)

        # low pass filter
        ft_low = low_pass(ft_shifted, 10)
        low_img = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_low)))

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()
        axs[0].imshow(block)
        axs[0].set_title("Original block")
        axs[1].imshow(gray, cmap='gray')
        axs[1].set_title("Gray block")
        axs[2].imshow(gray - low_img, cmap='gray')
        axs[2].set_title("Low frequencies subtracted")
        axs[3].imshow(img_filtered, cmap='gray')
        axs[3].set_title("High pass reconstruction")
        plt.tight_layout()
        # plt.savefig(f"cabinet_fly{i}.png")
        plt.show()

    # compare fruit fly blocks vs non fruit fly blocks in frequency space
    difference = fts[0] - fts[1]
    plt.imshow(ft_viz(difference))
    plt.show()


