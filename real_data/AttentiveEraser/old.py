from main import main
import torch
import os

def load_src_images(filename):
    filenames = os.listdir('examples/img/' + str(filename) +'/')
    sorted_files = sorted(filenames)
    print(len(sorted_files))
    return sorted_files

def load_mask_images(filename):
    filenames = os.listdir('examples/mask/' + str(filename) + "/")
    sorted_files = sorted(filenames)
    print(len(sorted_files))
    return sorted_files



if __name__ == "__main__":
    filename = "grapes"
    os.makedirs('outputs/' + str(filename), exist_ok=True)
    sorted_src_images = load_src_images(filename)
    sorted_mask_images = load_mask_images(filename)
    sorted_images = (sorted_src_images, sorted_mask_images)
    
    for i, (src_image, mask_image) in enumerate(zip(sorted_src_images, sorted_mask_images)):
        main("examples/img/" + str(filename) + '/' + str(src_image), "examples/mask/" + str(filename) + '/' + str(mask_image), f'outputs/{filename}/{filename}', 270, 480, i)

