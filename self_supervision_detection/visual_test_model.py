"""Function to visually test on real (out-of-sample) data"""

from PIL import Image
import matplotlib.pyplot as plt

image_list = [
        "../real_data/grapes_frames/grapes_0146.jpeg",
        "../real_data/grapes_frames/grapes_0147.jpeg",
        "../real_data/grapes_frames/grapes_0150.jpeg",
        "../real_data/mosquito_lab1_frames/mosquito_lab1_0008.jpeg",
        "../real_data/fruit_fly_wall_frames/fruit_fly_wall_0152.jpeg",
        "../real_data/fruit_fly_wall_frames/fruit_fly_wall_0038.jpeg"
        ]

def test_on_selected_images(model, transform, label):
    """Forward pass on selected images and save results"""
    model.eval()
    for i, image_path in enumerate(image_list):
        # load images
        img = Image.open(image_path)

        # forward pass on model
        model_input = transform(img)
        out = model(model_input)
        out = out.detach().squeeze().cpu().numpy()

        # save image and segmentation pair
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img)
        axs[1].imshow(out)
        axs[0].axis('off')
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(f"{label}_img{i}.png")

    model.train()
