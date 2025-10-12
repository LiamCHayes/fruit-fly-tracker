# Dataset Generation for Fly Segmentation

## How to generate data

0. Make sure you have numpy, PIL, and matplotlib available in your environment.
1. Put your background image in "dataset/backgrounds/\[your_image\].jpg". Make sure it is at least 6mp.
2. Run `pip install -e .` from the "data_generation" directory to install the utility functions.
3. Edit the make_data.py script in the main function to include the path to your image and how many datapoints you want to generate.
4. Run make_data.py from the "data_generation" directory.

## Dataset description

The dataset is composed of three folders:

**backgrounds** contains the background images with no synthetic flies superimposed. Each is a single jpg image named \[background_label\].png that corresponds to *multiple* datapoints in the images and masks directories. 

**images** contains the generated datapoint with the synthetic flies superimposed on the background. Each image is named \[background_label\]-\[datapoint_number\].jpg, corresponding to the single background image in the backgrounds directory.

**masks** contains the labels for the datapoint in the images directory. The naming convention is the same as images, except it is a png file instead of jpg. A 0 label means no fly and a 255 label means fly.

NOTE: the **images** and **masks** directories are omitted with a .gitignore file to save space on the repository. When we want to train, we can generate the dataset locally with the make_data.py script.
