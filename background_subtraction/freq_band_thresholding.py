"""Take the background subtraction and apply band pass filters to isolate the fly"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils import write_to_video, make_4_comparison, make_6_comparison

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-s", "--sequence",
            help="Sequence name (i.e. grapes_minpooled)")
    parser.add_argument(
            "-l", "--log-name",
            help="Name of the run (i.e. freq_attn)")
    parser.add_argument(
            "-n", "--name",
            help="Name of the output video")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Get background estimations
    dir = os.path.join(args.sequence, args.log_name, "background_estimation")
    frame_names = sorted(os.listdir(dir))
    background_paths = [os.path.join(dir, frame_name) for frame_name in frame_names]

    dir = os.path.join(args.sequence, "original_frames")
    frame_names = sorted(os.listdir(dir))
    actual_paths = [os.path.join(dir, frame_name) for frame_name in frame_names]

    video_frames = []
    for i in tqdm(range(len(background_paths))):
        background = Image.open(background_paths[i]).convert('L')
        background = np.array(background).astype(np.float32)

        actual = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR).convert('L')
        actual = np.array(actual).astype(np.float32)

        # Spatial domain subtraction and thresholding
        spatial_subtracted = (actual - background)**2
        spatial_subtracted_norm1 = np.sqrt(spatial_subtracted)
        spatial_subtracted_norm2 = spatial_subtracted / np.max(spatial_subtracted) * 255

        spatial_thresholded1 = spatial_subtracted_norm1 > 10
        spatial_thresholded2 = spatial_subtracted_norm1 > 20
        spatial_thresholded3 = spatial_subtracted_norm1 > 40
        spatial_thresholded4 = spatial_subtracted_norm1 > 60
        spatial_thresholded5 = spatial_subtracted_norm1 > 70
        spatial_thresholded6 = spatial_subtracted_norm1 > 80

        # Combine the frames
        # full_frame = make_4_comparison(actual, spatial_subtracted_norm1, spatial_thresholded4, spatial_subtracted_norm2)
        full_frame = make_6_comparison(spatial_thresholded1,
                                       spatial_thresholded2,
                                       spatial_thresholded3,
                                       spatial_thresholded4,
                                       spatial_thresholded5,
                                       spatial_thresholded6)
        video_frames.append(np.array(full_frame))

    output_dir = "output"
    write_to_video(video_frames, os.path.join(output_dir, f"{args.name}.mp4"), False)
