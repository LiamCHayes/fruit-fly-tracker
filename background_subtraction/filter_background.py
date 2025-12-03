"""Combine different levels of thresholding along with other priors to isolate the fruit fly across frames"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from utils import write_to_video, make_2_comparison, make_4_comparison, make_6_comparison

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

def write_to_video(frames, file_path, color=True):
    out = cv2.VideoWriter(file_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          30, 
                          (frames[0].shape[1], frames[0].shape[0]), 
                          isColor=color)

    # frames = [frame.astype(np.uint8) * 255 for frame in frames]
    for frame in frames:
        # color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved to {file_path}!")

def get_thresholds(subtracted_frame):
    spatial_thresholded1 = subtracted_frame > 10
    spatial_thresholded2 = subtracted_frame > 20
    spatial_thresholded3 = subtracted_frame > 30
    spatial_thresholded4 = subtracted_frame > 40
    spatial_thresholded5 = subtracted_frame > 50
    spatial_thresholded6 = subtracted_frame > 60
    spatial_thresholded7 = subtracted_frame > 70
    spatial_thresholded8 = subtracted_frame > 80
    spatial_thresholded9 = subtracted_frame > 90
    spatial_thresholded10 = subtracted_frame > 100

    return [spatial_thresholded1,
            spatial_thresholded2,
            spatial_thresholded3,
            spatial_thresholded4,
            spatial_thresholded5,
            spatial_thresholded6,
            spatial_thresholded7,
            spatial_thresholded8,
            spatial_thresholded9,
            spatial_thresholded10
            ]

def subtract_and_filter(actual, background):
    # Spatial domain subtraction and thresholding
    spatial_subtracted = (actual - background)**2
    spatial_subtracted_norm1 = np.sqrt(spatial_subtracted)

    thresholds = get_thresholds(spatial_subtracted_norm1)

    # Median filter on the thresholds
    filtered_thresholds = []
    for threshold in thresholds:
        filtered_thresholds.append(ndimage.median_filter(threshold.astype(np.float32), size=3))

    # Edge detection
    blurred_background = cv2.GaussianBlur(background, (7, 7), 0)
    sobel_h = ndimage.sobel(blurred_background, axis=0)
    sobel_v = ndimage.sobel(blurred_background, axis=1)
    magnitude = np.hypot(sobel_h, sobel_v)
    magnitude[magnitude < 100] = 0
    magnitude *= 255.0 / np.max(magnitude)
    blurred_magnitude = cv2.GaussianBlur(magnitude, (11, 11), 0)

    # Frame averaging
    min_level = 10
    activation_bool = [np.sum(t) > min_level for t in filtered_thresholds]
    max_conf_threshold = np.max([0, np.sum(activation_bool)])

    stacked_thresholds = np.stack(filtered_thresholds[0:max_conf_threshold])
    summed_thresholds = np.sum(stacked_thresholds, axis=0)
    summed_thresholds = summed_thresholds - blurred_magnitude / 5
    averaged_thresholds = summed_thresholds / max_conf_threshold

    # Post-process the averaged frame
    scaling_factor = 4
    averaged_thresholds[averaged_thresholds < 0] = 0
    averaged_thresholds = averaged_thresholds ** scaling_factor
    averaged_thresholds = averaged_thresholds / np.max(averaged_thresholds) * 255

    final_thresholded = averaged_thresholds > 200
    final_thresholded = final_thresholded.astype(np.uint8) * 255

    # Group detections with cv2 contours
    contours, _ = cv2.findContours(final_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        # Get centroid
        moments = cv2.moments(contour)
        if moments["m00"] !=0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            centers.append(np.array([cX, cY]))
        else:
            centers.append(np.array([contour[0, 0, 0], contour[0, 0, 1]]))

    return centers

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
    threshold_sums = []
    for i in tqdm(range(len(background_paths))):
        background = Image.open(background_paths[i]).convert('L')
        background = np.array(background).astype(np.float32)

        actual_color = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR)
        actual_color = cv2.cvtColor(np.array(actual_color), cv2.COLOR_RGB2BGR)
        actual = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR).convert('L')
        actual = np.array(actual).astype(np.float32)

        # Spatial domain subtraction and thresholding
        spatial_subtracted = (actual - background)**2
        spatial_subtracted_norm1 = np.sqrt(spatial_subtracted)
        spatial_subtracted_norm2 = spatial_subtracted / np.max(spatial_subtracted) * 255

        thresholds = get_thresholds(spatial_subtracted_norm1)

        # Median filter on the thresholds
        filtered_thresholds = []
        for i, threshold in enumerate(thresholds):
            filtered_thresholds.append(ndimage.median_filter(threshold.astype(np.float32), size=3))

        threshold_sum = [np.sum(t) for t in filtered_thresholds]
        threshold_sums.append(threshold_sum)

        # full_frame = make_6_comparison(filtered_thresholds[0],
                                       # filtered_thresholds[1],
                                       # filtered_thresholds[2],
                                       # filtered_thresholds[3],
                                       # filtered_thresholds[4],
                                       # filtered_thresholds[5])
        # full_frame = full_frame * 255
        # full_frame = full_frame.astype(np.uint8)
        # video_frames.append(full_frame)

        # Edge detection
        blurred_background = cv2.GaussianBlur(background, (7, 7), 0)
        sobel_h = ndimage.sobel(blurred_background, axis=0)
        sobel_v = ndimage.sobel(blurred_background, axis=1)
        magnitude = np.hypot(sobel_h, sobel_v)
        magnitude[magnitude < 100] = 0
        magnitude *= 255.0 / np.max(magnitude)
        blurred_magnitude = cv2.GaussianBlur(magnitude, (11, 11), 0)

        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(background, cmap="gray")
        # axs[1].imshow(magnitude, cmap="gray")
        # plt.show()

        # Frame averaging
        min_level = 10
        activation_bool = [np.sum(t) > min_level for t in filtered_thresholds]
        max_conf_threshold = np.max([0, np.sum(activation_bool)])

        stacked_thresholds = np.stack(filtered_thresholds[0:max_conf_threshold])
        summed_thresholds = np.sum(stacked_thresholds, axis=0)
        summed_thresholds = summed_thresholds - blurred_magnitude / 5
        averaged_thresholds = summed_thresholds / max_conf_threshold

        # Post-process the averaged frame
        scaling_factor = 4
        averaged_thresholds[averaged_thresholds < 0] = 0
        averaged_thresholds = averaged_thresholds ** scaling_factor
        averaged_thresholds = averaged_thresholds / np.max(averaged_thresholds) * 255

        final_thresholded = averaged_thresholds > 200
        final_thresholded = final_thresholded.astype(np.uint8) * 255
        final_thresholded_color = cv2.cvtColor(final_thresholded, cv2.COLOR_GRAY2BGR)

        full_frame = make_2_comparison(actual_color, final_thresholded_color)
        video_frames.append(full_frame)

        # Group detections with cv2 contours
        contours, _ = cv2.findContours(final_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        epsilon = 0.00001
        for contour in contours:
            # Get centroid
            moments = cv2.moments(contour)
            if moments["m00"] !=0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                centers.append(np.array([cX, cY]))
            else:
                centers.append(np.array([contour[0, 0, 0], contour[0, 0, 1]]))

        contour_img = cv2.cvtColor(final_thresholded, cv2.COLOR_GRAY2BGR)
        for c in centers:
            cv2.circle(contour_img, (c[0], c[1]), 4, (0, 0, 255), -1)
        full_frame = make_2_comparison(actual_color, contour_img)
        video_frames.append(full_frame)

    # Write video
    output_dir = "output"
    write_to_video(video_frames, os.path.join(output_dir, f"{args.name}.mp4"), True)

    # Save threshold sum plot
    # frame_num = np.arange(len(video_frames))
    # sum_of_levels = list(zip(*threshold_sums))
    # fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    # axs = axs.flatten()
    # axs[0].plot(frame_num, sum_of_levels[0])
    # axs[1].plot(frame_num, sum_of_levels[1])
    # axs[2].plot(frame_num, sum_of_levels[2])
    # axs[3].plot(frame_num, sum_of_levels[3])
    # axs[4].plot(frame_num, sum_of_levels[4])
    # axs[5].plot(frame_num, sum_of_levels[5])
    # axs[6].plot(frame_num, sum_of_levels[6])
    # axs[7].plot(frame_num, sum_of_levels[7])
    # axs[8].plot(frame_num, sum_of_levels[8])
# 
    # axs[0].set_title("Threshold 10")
    # axs[1].set_title("Threshold 20")
    # axs[2].set_title("Threshold 30")
    # axs[3].set_title("Threshold 40")
    # axs[4].set_title("Threshold 50")
    # axs[5].set_title("Threshold 60")
    # axs[6].set_title("Threshold 70")
    # axs[7].set_title("Threshold 80")
    # axs[8].set_title("Threshold 90")
# 
    # custom_ylim = (0, np.max(sum_of_levels[0]))
    # plt.setp(axs, ylim=custom_ylim)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "activation_levels.png"))
    # plt.close()
