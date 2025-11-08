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

def ft_viz(ft):
    return np.log(np.abs(ft) + 1)

if __name__ == "__main__":
    image = Image.open('../real_data/grapes_frames/grapes_0147.jpeg')
    image_gray = image.convert('L')
    data = np.array(image_gray, dtype=float)

    # Compute 2D FFT of image
    ft = np.fft.fft2(data)
    ft_shifted = np.fft.fftshift(ft)

    filtered_ft = high_pass(ft_shifted)
    ft_ifft_shfited = np.fft.ifftshift(filtered_ft)
    img_filtered = np.fft.ifft2(ft_ifft_shfited)
    img_filtered = np.log(np.abs(img_filtered))

    # Show the result
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs = axs.flatten()
    axs[0].imshow(image)
    axs[1].imshow(img_filtered, cmap='gray')
    axs[2].imshow(ft_viz(ft_shifted))
    axs[3].imshow(ft_viz(filtered_ft))
    plt.show()

