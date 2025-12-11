import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to folder of frames to minsample")
    args = parser.parse_args()
    arg_path = args.path
    
    path_origin = os.getcwd()
    path = os.getcwd() +"/" + os.path.join(arg_path)
    os.makedirs(path + "/upsampled_restored" + str(arg_path), exist_ok=True)
    for filename in os.listdir(path):
        if filename.lower().endswith((".jpg", ".jpeg")):
            full_path = os.path.join(path, filename)
            
            img = Image.open(full_path)
            resized = ImageOps.fit(
                    img,
                    (512, 256),
                    method=Image.BICUBIC
                    )
            
            image = np.array(resized)
            
            img_pil = Image.fromarray(image)
            img_pil.save(path +"/upsampled_restored"+ str(arg_path) +"/" + filename.lower())
