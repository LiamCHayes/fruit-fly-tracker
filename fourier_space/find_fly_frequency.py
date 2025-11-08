"""Visualize frequency space of a fruit fly"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# Block images
class BlockCoords:
    def __init__(self, row_start, row_stop, col_start, col_stop):
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop

    def get_center(self, max_row):
        return (self.col_start + self.col_stop) / 2, max_row - (self.row_start + self.row_stop) / 2

    def shift(self, row_shift, col_shift):
        row_start = self.row_start + row_shift
        row_stop = self.row_stop + row_shift
        col_start = self.col_start + col_shift
        col_stop = self.col_stop + col_shift

        shifted = BlockCoords(row_start, row_stop, col_start, col_stop)

        return shifted

    def __sub__(self, other):
        if isinstance(other, BlockCoords):
            return (self.row_start - other.row_start, self.col_start - other.col_start)
        else:
            raise TypeError("not supported")

    def __str__(self):
        return f"{self.row_start} {self.row_stop}\n{self.col_start} {self.col_stop}"

def get_blocks(image, block_size):
    """returns a list of blocks in row-major format"""
    H, W, C = image.shape
    n_h_tiles = int(H / block_size)
    n_w_tiles = int(W / block_size)
    coords = []
    tiles = []
    for row in range(n_h_tiles):
        for col in range(n_w_tiles):
            row_start = row * block_size
            row_stop = (row+1) * block_size
            col_start = col * block_size
            col_stop = (col+1) * block_size
            coords.append(BlockCoords(row_start, row_stop, col_start, col_stop))
            tiles.append(image[row_start:row_stop, col_start:col_stop, :])
    return tiles, coords

def ft_viz(ft):
    return np.log(np.abs(ft) + 1)

if __name__ == "__main__":
    fly_seq = "cabinet"

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
        axs[3].imshow(high_filtered, cmap='gray')
        plt.tight_layout()
        plt.savefig(f"cabinet_fly{i}.png")
        plt.show()

    # compare fruit fly blocks vs non fruit fly blocks in frequency space
    difference = fts[0] - fts[1]
    plt.imshow(ft_viz(difference))
    plt.show()


