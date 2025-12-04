import cv2
import numpy as np

def write_to_video(frames, file_path, color=True):
    out = cv2.VideoWriter(file_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          10, 
                          (frames[0].shape[1], frames[0].shape[0]), 
                          isColor=color)

    # frames = [frame.astype(np.uint8) * 255 for frame in frames]
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {file_path}!")

def make_2_comparison(right, left):
    full_frame = np.concatenate([right, left], axis=1).astype(np.uint8)

    return full_frame

def make_4_comparison(top_left, top_right, bottom_left, bottom_right):
    top_row = np.concatenate([top_left, top_right], axis=1)
    bottom_row = np.concatenate([bottom_left, bottom_right], axis=1)
    full_frame = np.concatenate([top_row, bottom_row], axis=0).astype(np.uint8)

    return full_frame

def make_6_comparison(frame_1, frame_2, frame_3, frame_4, frame_5, frame_6):
    top_row = np.concatenate([frame_1, frame_2, frame_3], axis=1)
    bottom_row = np.concatenate([frame_4, frame_5, frame_6], axis=1)
    full_frame = np.concatenate([top_row, bottom_row], axis=0).astype(np.uint8)

    return full_frame

def make_2_comparison_color(right, left):
    full_frame = np.concatenate([right, left], axis=1).astype(np.uint8)

    return full_frame
