import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import os

def min_pool(image, num_downsamples=1):
    image = torch.from_numpy(image)
    image = image.permute(2,1,0)
    img_shape = image.shape

    image = -image.float()
    for i in range(0, num_downsamples):
        output = F.max_pool2d(image, kernel_size=2, stride=2)
        image = output

    image = (torch.abs(-image))
    image = image.numpy().astype(np.uint8)
    image = np.transpose(image, (2,1,0))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to folder of frames to minsample")
    args = parser.parse_args()
    arg_path = args.path
    
    path_origin = os.getcwd()
    path = os.getcwd() +"/" + os.path.join(arg_path)
    os.makedirs(path + "/minpooled", exist_ok=True)
    for filename in os.listdir(path):
        if filename.lower().endswith((".jpg", ".jpeg")):
            full_path = os.path.join(path, filename)
            
            img = Image.open(full_path)
            image = np.array(img)
            image_min = min_pool(image, 2)
            img_pil = Image.fromarray(image_min)
            img_pil.save(path +"/minpooled/" + filename.lower())
