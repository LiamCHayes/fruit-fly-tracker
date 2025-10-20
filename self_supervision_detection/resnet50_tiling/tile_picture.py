"""Class to tile an input image to 224x224 and re-stitch model output"""

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ImageTiler:
    """Tiles and stitches images and model output"""
    def __init__(self, tile_size=224):
        self.padded_img_size = None
        self.orig_img_size = None
        self.n_tiles = None
        self.pad_h = None
        self.pad_w = None
        self.tile_size = tile_size

    def tile_image(self, img: torch.Tensor, batched = False):
        """Tiles image"""
        batched_dim = -1 + int(batched)
        channel_dim = 0 + int(batched)
        height_dim = 1 + int(batched)
        width_dim = 2 + int(batched)

        # pad images so we get even tiles
        H, W = img.shape[-2:]
        self.orig_img_size = img.shape
        self.pad_h = (self.tile_size - H % self.tile_size) % self.tile_size
        self.pad_w = (self.tile_size - W % self.tile_size) % self.tile_size
        padded = F.pad(img, (0, self.pad_w, 0, self.pad_h), mode='constant', value=0)
        self.padded_img_size = padded.shape

        # tile the image
        tiles_h = padded.unfold(height_dim, self.tile_size, self.tile_size)
        tiles = tiles_h.unfold(width_dim, self.tile_size, self.tile_size)

        if batched:
            tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
            tiles = tiles.view(-1, img.size(channel_dim), self.tile_size, self.tile_size)
            self.n_tiles = int(tiles.size(0) / self.orig_img_size[0])
        else:
            tiles = tiles.contiguous().view(img.size(channel_dim), -1, self.tile_size, self.tile_size)
            tiles = tiles.permute(1, 0, 2, 3)
            self.n_tiles = tiles.size(0)

        return tiles

    def stitch_tiles(self, tiles: torch.Tensor, batched=False):
        """Stitch tiles back together"""
        if self.padded_img_size is None:
            raise ValueError("Call tile_image before stitch_tiles")

        fold = nn.Fold(output_size=self.padded_img_size[-2:], kernel_size=self.tile_size, stride=self.tile_size)

        if batched:
            C = tiles.size(1)
            B = int(tiles.size(0) / self.n_tiles)
            tiles_reshaped = tiles.contiguous().view(B, self.n_tiles, C, self.tile_size, self.tile_size)
            tiles_reshaped = tiles_reshaped.permute(0, 2, 3, 4, 1).contiguous()
            tiles_reshaped = tiles_reshaped.view(B, C * self.tile_size * self.tile_size, self.n_tiles)
            reconstructed = fold(tiles_reshaped)
        else:
            tiles_reshaped = tiles.contiguous().view(tiles.size(0), -1).T
            reconstructed = fold(tiles_reshaped.unsqueeze(0))

        if self.pad_h > 0 or self.pad_w > 0:
            reconstructed = reconstructed[:, :, :-self.pad_h if self.pad_h > 0 else None,
                                          :-self.pad_w if self.pad_w > 0 else None]

        return reconstructed


if __name__ == "__main__":
    # debug settings
    show_tiles = True
    show_reconstructed = True

    img1 = Image.open("../../data_generation/dataset/train/images/windowsill-00999.jpeg")
    img2 = Image.open("../../data_generation/dataset/train/images/counter_with_pen-00288.jpeg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        ])
    img_tensor1 = preprocess(img1)
    img_tensor2 = preprocess(img2)
    img_tensor = torch.stack([img_tensor1, img_tensor2], dim=0)

    tiler = ImageTiler()
    tiles = tiler.tile_image(img_tensor, batched=True)
    reconstructed = tiler.stitch_tiles(tiles, batched=True)

    # Show reconstructed
    if show_reconstructed:
        import matplotlib.pyplot as plt
        for img_idx in range(reconstructed.size(0)):
            img = reconstructed[img_idx, :, :, :]
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.show()

    # Show tiles
    if show_tiles:
        import matplotlib.pyplot as plt
        for tile_idx in range(tiles.size(0)):
            tile = tiles[tile_idx, :, :, :].permute(1, 2, 0)
            plt.imshow(tile)
            plt.show()

