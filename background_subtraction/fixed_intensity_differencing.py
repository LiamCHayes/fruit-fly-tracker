"""Does intensity differencing on a video with a fixed frame"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from utils import write_to_video, make_2_comparison

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

    dir = os.path.join(args.sequence, "original_frames")
    frame_names = sorted(os.listdir(dir))
    actual_paths = [os.path.join(dir, frame_name) for frame_name in frame_names]

    video_frames = []
    prev_frame = None
    background = []
    for i in tqdm(range(len(actual_paths))):
        actual_color = Image.open(actual_paths[i])
        actual_color = cv2.cvtColor(np.array(actual_color), cv2.COLOR_RGB2BGR)
        actual = Image.open(actual_paths[i]).convert('L')
        actual = np.array(actual).astype(np.float32)

        if i == 0:
            prev_frame = actual
            background.append(actual)
            continue

        # Subtract previous frame from current frame
        background_array = np.stack(background, axis=0)
        background_median = np.median(background_array, axis=0)

        subtracted = (actual - prev_frame)**2
        background_subtracted = (actual - background_median)**2

        subtracted = np.sqrt(subtracted)
        background_subtracted = np.sqrt(background_subtracted)

        prev_frame = actual
        background.append(actual)

        # Make video
        subtracted_img = cv2.cvtColor(background_subtracted, cv2.COLOR_GRAY2BGR)
        full_frame = make_2_comparison(actual_color, subtracted_img)
        video_frames.append(full_frame)

    # Write video
    output_dir = "output"
    write_to_video(video_frames, os.path.join(output_dir, f"{args.name}.mp4"), True)
