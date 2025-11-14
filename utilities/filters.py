"""Defines a high pass filter in frequency domain"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def high_pass(ft_shifted, radius=30):
    # Create a high-pass filter mask
    rows, cols = ft_shifted.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    r = radius  # radius of low frequency to block
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0

    # Apply mask
    ft_filtered = ft_shifted * mask

    return ft_filtered

def low_pass(ft_shifted, cutoff=30):
    # Create a low-pass filter mask
    rows, cols = ft_shifted.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    c = cutoff
    mask[crow - c:crow + c, ccol - c:ccol + c] = 1

    # Apply mask
    ft_filtered = ft_shifted * mask

    return ft_filtered

def ft_viz(ft):
    return np.log(np.abs(ft) + 1)

