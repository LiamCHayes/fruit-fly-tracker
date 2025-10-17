"""Evaluate detection models on real-world data"""

import numpy as np
import cv2
import argparse
import os

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of the .mp4 file (without the .mp4)")
args = parser.parse_args()

# File path things
video_name = args.name
video_path = f"../real_data/{video_name}.mp4"
frames_path = f"{video_name}_frames/"
if not os.path.isdir(frames_path):
    os.makedirs(frames_path)

# Read frames and save
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[Error] could not open video {video_path}")
    exit()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    frame_filename = f"{video_name}_{frame_count:04d}.jpeg"
    cv2.imwrite(frames_path + frame_filename, frame)
    if frame_count > 999:
        print("[INFO] Video too long, saved 999 frames")
        break
cap.release()
