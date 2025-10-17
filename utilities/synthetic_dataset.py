"""Torch Dataset class for the synthetic dataset"""

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SyntheticData(Dataset):
    def __init__(self, image_dir, label_dir, background_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir 
        self.background_dir = background_dir
        
        self.images = os.listdir(self.image_dir)
        self.labels = os.listdir(self.label_dir)

        self.label_transform = transforms.ToTensor()
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.user_transforms = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.image_dir + self.images[idx]) as img:
            sample = self.img_transform(img)
        with Image.open(self.label_dir + self.labels[idx]) as mask:
            label = self.label_transform(np.array(mask))

        if self.user_transforms is not None:
            sample = self.user_transforms(sample)

        return sample, label

if __name__ == "__main__":
    image_dir = "../data_generation/dataset/train/images/"
    label_dir = "../data_generation/dataset/train/masks/"
    background_dir = "../data_generation/dataset/backgrounds/"

    dataset = SyntheticData(image_dir, label_dir, background_dir)

    img, label = dataset.__getitem__(0)
    print(img.shape, label.shape)
