"""Torch module that extracts the frequency features from an image"""

import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
from utilities.image_utils import get_blocks

def zigzag_flatten(dct_block):
    """ZigZag flattens an array so that the frequencies from the dct block are in a 1 dimensional array from low to high"""
    if dct_block.ndim == 3 and dct_block.shape[2] == 1:
        dct_block = np.squeeze(dct_block, axis=2)

    if dct_block.shape != (8, 8):
        raise ValueError("ZigZag Shape Error")

    rows, cols = dct_block.shape
    flattened = []

    for diagonal_sum in range(rows + cols - 1):
        # Determine the start and end indices for this diagonal
        if diagonal_sum < rows:
            start_row = 0
            end_row = diagonal_sum + 1
        else:
            start_row = diagonal_sum - cols + 1
            end_row = rows

        # Extract elements along the diagonal
        diagonal_elements = []
        for i in range(start_row, end_row):
            j = diagonal_sum - i
            if 0 <= i < rows and 0 <= j < cols:
                diagonal_elements.append(dct_block[i, j])

        # Reverse the diagonal elements if the diagonal sum is even
        if diagonal_sum % 2 == 0:
            flattened.extend(diagonal_elements[::-1])
        else:
            flattened.extend(diagonal_elements)

    return np.array(flattened)

class SpectrumAttention(nn.Module):
    def __init__(self, num_heads, spatial_embed_dim=64, frequency_embed_dim=2048, dropout=0.1):
        super(SpectrumAttention, self).__init__()

        # Multi Head Attention
        self.spatial_attention = nn.MultiheadAttention(
                embed_dim=spatial_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True)
        self.spatial_norm1 = nn.LayerNorm(spatial_embed_dim)
        self.spatial_norm2 = nn.LayerNorm(spatial_embed_dim)
        self.spatial_dropout = nn.Dropout(dropout)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_embed_dim, spatial_embed_dim*2),
            nn.GELU(),
            nn.Linear(spatial_embed_dim*2, spatial_embed_dim),
            nn.Dropout(dropout))

        self.frequency_attention = nn.MultiheadAttention(
                embed_dim=frequency_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True)
        self.frequency_norm1 = nn.LayerNorm(frequency_embed_dim)
        self.frequencyspatial_norm2 = nn.LayerNorm(frequency_embed_dim)
        self.frequencyspatial_dropout = nn.Dropout(dropout)
        self.frequencyspatial_mlp = nn.Sequential(
            nn.Linear(frequency_embed_dim, frequency_embed_dim*2),
            nn.GELU(),
            nn.Linear(frequency_embed_dim*2, frequency_embed_dim),
            nn.Dropout(dropout))

        # Convolutions
        self.spatial_convolution = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1)
        self.frequency_convolution = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1)
        self.final_convolution = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0)
        # self.ds_conv_block TODO

    def forward(self, spatial_tokens, frequency_tokens):
        """
        spatial_tokens is of dimension B, 2048, 64
        frequency_tokens is of dimension B, 64, 2048
        """
        # Spatial attention
        spatial_attn_input = self.spatial_norm1(spatial_tokens)
        spatial_attn_output = spatial_tokens + self.spatial_attention(spatial_attn_input)
        spatial_attn_output = self.spatial_dropout(spatial_attn_output)
        spatial_attn_output = self.spatial_norm2(spatial_attn_output)
        spatial_attn_output = spatial_tokens + self.spatial_mlp(spatial_attn_output)

        # Frequency attention
        frequency_attn_input = self.frequency_norm1(frequency_tokens)
        frequency_attn_output = frequency_tokens + self.frequency_attention(frequency_attn_input)
        frequency_attn_output = self.frequency_dropout(frequency_attn_output)
        frequency_attn_output = self.frequency_norm2(frequency_attn_output)
        frequency_attn_output = frequency_tokens + self.frequency_mlp(frequency_attn_output)

        # Convolutions
        input(f"Spatial attention output size {spatial_attn_output.shape}")
        input(f"Frequency attention output size {frequency_attn_output.shape}")
        # TODO reshape attention outputs for convolutions
        spatial_convolved = self.spatial_convolution(spatial_output_reshaped)
        frequency_convolved = self.frequency_convolution(frequency_output_reshaped)
        combined = spatial_convolved + frequency_convolved
        final_convolved = self.final_convolution(combined)

        return final_convolved

if __name__ == "__main__":
    fp = "../real_data/background_estimation_desk/frame000000.png"
    img_rgb = Image.open(fp)

    # Split image into YCbCr
    img_ycbcr = img_rgb.convert('YCbCr')
    Y, Cb, Cr = img_ycbcr.split()

    # Get the shape of the image
    y_channel = np.array(Y)
    h_prime = y_channel.shape[0] / 8
    w_prime = y_channel.shape[1] / 8

    # Get the 8x8 blocks
    tiles, coords = get_blocks(np.expand_dims(y_channel, axis=-1), 8)

    # DCT of each tile
    tokens = []
    for tile in tiles:
        y_prime = dct(np.squeeze(tile), type=2, norm='ortho')
        tokens.append(zigzag_flatten(y_prime))

    # Get the frequency and spatial tokens
    token_array = np.array(tokens)
    spatial_tokens = np.split(token_array, 2048, axis=0)
    frequency_tokens = np.split(token_array, 64, axis=1)

    tokens_shape = (int(h_prime), int(w_prime), 64)
    token_array = token_array.reshape(tokens_shape)
    print(token_array.shape)
