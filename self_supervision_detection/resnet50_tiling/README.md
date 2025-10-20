# ResNet50 encoder with tiling

Tiles the images to fit ResNet50 input size, performs self-supervised learning, then gets fine tuned on synthetic data.

## Log

Iteration 1:
- Self supervison with image reconstruction, L1 loss per tile, Adam optimizer, input is only tiled image (no downsampled full image)

Future:
- Add perceptual loss
- add a downsampled version of the whole image to the tiles for global context
- loss on the entire image reconstruction instead of per-tile
