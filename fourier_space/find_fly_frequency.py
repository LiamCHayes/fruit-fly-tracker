"""Visualize frequency space of a fruit fly"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from high_pass_filter import high_pass
from utilities.image_utils import get_blocks, ft_viz

if __name__ == "__main__":
    fly_seq = "grape"

    if fly_seq == "grape":
        dir_path = "../real_data/grapes_frames/"
        block_idxs = [69, 72]
    elif fly_seq == "cabinet":
        dir_path = "../real_data/fruit_fly_cabinet_frames/"
        block_idxs = [80, 81, 82]
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

    # get blocks
    blocks, block_coords = get_blocks(frame, 256)

    # loop through important blocks
    fts = []
    for i, block in enumerate(blocks):
        if i not in block_idxs:
            continue
        print(i)

        # fourier transform of blocks
        gray = np.mean(block, axis=2)
        ft = np.fft.fft2(gray)
        ft_shifted = np.fft.fftshift(ft)
        fts.append(ft_shifted)

        # high pass filter frequency space
        ft_filtered = high_pass(ft_shifted, 10)
        ft_ifft_shfited = np.fft.ifftshift(ft_filtered)
        img_filtered = np.fft.ifft2(ft_ifft_shfited)
        img_filtered = np.abs(img_filtered)

        # high pass filter
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        high_filtered = ndimage.convolve(gray, kernel)

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()
        axs[0].imshow(frame)
        axs[1].imshow(block)
        axs[2].imshow(ft_viz(ft_shifted))
        axs[3].imshow(ft_viz(ft_filtered))
        plt.tight_layout()
        # plt.savefig(f"cabinet_fly{i}.png")
        plt.show()

    # compare fruit fly blocks vs non fruit fly blocks in frequency space
    difference = fts[0] - fts[1]
    plt.imshow(ft_viz(difference))
    plt.show()


