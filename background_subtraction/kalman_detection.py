"""Combine the filtered threshold with a kalman filter to get trajectories"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, euclidean
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import IMMEstimator as IMM
from filterpy.kalman import MerweScaledSigmaPoints
from filter_background import subtract_and_filter

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
            "-m", "--max-association-cost",
            help="Association threshold to keep a detection as part of a kalman filter track")
    parser.add_argument(
            "-n", "--name",
            help="Name of the output video")

    return parser.parse_args()

# Kalman filter transition functions
def f1(x, dt):
    """State transition function for constant velocity"""
    vx = x[2] * np.cos(x[3])
    vy = x[2] * np.sin(x[3])

    x_next = x.copy()
    x_next[0] += vx * dt
    x_next[1] += vy * dt

    return x_next

def f2(x, dt, a):
    """State transition function for constant acceleration"""
    v = x[2]
    theta = x[3]

    x_next = x.copy()
    x_next[0] += (v * dt + 0.5 * a * dt**2) * np.cos(theta)
    x_next[1] += (v * dt + 0.5 * a * dt**2) * np.sin(theta)
    x_next[2] += a * dt

    return x_next

def f3(x, dt, omega):
    """State transition function for constant angular velocity"""
    v = x[2]
    theta = x[3]

    x_next = x.copy()
    if abs(omega) > 0.001:
        x_next[0] += (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        x_next[1] += (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
    else:
        x_next[0] += v * dt * np.cos(theta)
        x_next[1] += v * dt * np.sin(theta)

    x_next[3] = theta + omega * dt

    return x_next

def h(x):
    """Measurement function"""
    return [x[0], x[1]]

def get_kalman_IMM(initial_state):
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=-1)

    ukf_cv = UKF(dim_x=4, dim_z=2, dt=1, fx=f1, hx=h, points=points) # Constant velocity kalman filter
    ukf_ca = UKF(dim_x=4, dim_z=2, dt=1, fx=f2, hx=h, points=points) # Constant acceleration
    ukf_cw = UKF(dim_x=4, dim_z=2, dt=1, fx=f3, hx=h, points=points) # Constant angular velocity (omega)

    ukfs = [ukf_cv, ukf_ca, ukf_cw]
    for ukf in ukfs:
        ukf.x = initial_state

    mu = [1 / 3, 1 / 3, 1 / 3]
    trans = np.ones((3, 3)) / 9
    imm = IMM(ukfs, mu, trans)

    return imm

def fourier_image_displacement(frame_1, frame_2):
    """Gets the top 3 displacement vectors for the whole image"""
    # compute fourier transform of each image
    ft_1 = np.fft.fft2(frame_1)
    ft_2 = np.fft.fft2(frame_2)

    # compute the fourier transform of the cross correlation
    psi = ft_1.conj() * ft_2
    psi_tilde = psi / np.maximum(np.abs(psi), 0.001)

    # inverse fourier transform
    R_tilde = np.fft.ifft2(psi_tilde)

    # find local maxima
    filtered = maximum_filter(R_tilde.real, size=5)
    maxima_mask = (R_tilde.real == filtered)
    displacements = np.argwhere(maxima_mask)
    row, col = list(zip(*displacements))
    max_values = []
    for point_idx in range(len(row)):
        max_values.append(R_tilde[row[point_idx], col[point_idx]].real)
    sorted_pairs = sorted(zip(displacements, max_values), key=lambda x:x[1], reverse=True)
    top_3_displacements = [idx for idx, _ in sorted_pairs[:3]]

    return top_3_displacements

if __name__ == "__main__":
    args = get_args()

    # Get background estimations
    dir = os.path.join(args.sequence, args.log_name, "background_estimation")
    frame_names = sorted(os.listdir(dir))
    background_paths = [os.path.join(dir, frame_name) for frame_name in frame_names]

    dir = os.path.join(args.sequence, "original_frames")
    frame_names = sorted(os.listdir(dir))
    actual_paths = [os.path.join(dir, frame_name) for frame_name in frame_names]

    prev_background = None
    IMMs = []
    next_frame_preds = []
    video_frames = []
    for i in tqdm(range(len(background_paths))):
        background = Image.open(background_paths[i]).convert('L')
        background = np.array(background).astype(np.float32)

        actual_color = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR)
        actual_color = cv2.cvtColor(np.array(actual_color), cv2.COLOR_RGB2BGR)
        actual = Image.open(actual_paths[i]).resize((512, 256), resample=Image.BILINEAR).convert('L')
        actual = np.array(actual).astype(np.float32)

        # Get fly predictions
        fly_predictions = subtract_and_filter(actual, background)

        # Kalman filter the fly predictions
        next_frame_preds = [h(pred) for pred in next_frame_preds]

        n_current_tracks = len(next_frame_preds)
        n_detections = len(fly_predictions)
        new_detections = []
        if n_current_tracks != 0 and n_detections != 0:
            # Compute cost matrix
            tracks_coords = np.array(next_frame_preds)
            detection_coords = np.array(fly_predictions)
            cost_matrix = cdist(tracks_coords, detection_coords, metric='euclidean')

            # Assign past predictions to current detections
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Discard high-cost assignments
            assigned_tracks = []
            unassigned_tracks = list(range(cost_matrix.shape[0]))
            unassigned_detections = list(range(cost_matrix.shape[1]))

            for i, track_idx in enumerate(row_ind):
                detection_idx = col_ind[i]
                cost = cost_matrix[track_idx, detection_idx]
                if cost < np.float64(args.max_association_cost):
                    assigned_tracks.append((track_idx, detection_idx))

                    if track_idx in unassigned_tracks:
                        unassigned_tracks.remove(track_idx)
                    if detection_idx in unassigned_detections:
                        unassigned_detections.remove(detection_idx)

            # Update existing tracks with the new associated detection
            for assignment in assigned_tracks:
                new_detection = fly_predictions[assignment[1]]
                new_detections.append(new_detection)
                IMMs[assignment[0]].update(new_detection)

            # Create new tracks for unassigned detections
            for unassigned_detection_idx in unassigned_detections:
                prediction = fly_predictions[unassigned_detection_idx]
                initial_state = np.array([prediction[0], prediction[1], 0, 0])
                IMMs.append(get_kalman_IMM(initial_state))

            # Delete tracks that did not find a matching detection
            for unassigned_track_idx in reversed(sorted(unassigned_tracks)):
                IMMs.pop(unassigned_track_idx)

        elif n_current_tracks == 0 and n_detections != 0:
            # Make a new track for all detections
            for prediction in fly_predictions:
                initial_state = np.array([prediction[0], prediction[1], 0, 0])
                IMMs.append(get_kalman_IMM(initial_state))
        elif n_current_tracks != 0 and n_detections == 0:
            # TODO Count how many times this happens and set new detection as prediction?
            continue
        else:
            # No tracks and no detections, wait until we get a detection
            continue

        # Predict next location of all detections using kalman filter
        next_frame_preds = []
        for imm in IMMs:
            # Temporary functions for argument passing
            ACCELERATION = 0.01
            OMEGA = 0.05
            DT = 1.0
            def f2_bound(x, dt):
                return f2(x, dt, ACCELERATION)
            def f3_bound(x, dt):
                return f3(x, dt, OMEGA)

            imm.filters[0].fx = f1
            imm.filters[1].fx = f2_bound
            imm.filters[2].fx = f3_bound
            for f in imm.filters:
                f.dt = DT

            # Predict next locations using kalman filters
            imm.predict()
            next_frame_preds.append(imm.x.T)

        # Visualize detections
        kalman_filter_output = np.zeros_like(actual_color)

        # draw circles of the kalman filtered detections
        for detection in new_detections:
            cv2.circle(kalman_filter_output, (detection[0], detection[1]), 4, (0, 0, 255), -1)

        # draw smaller circles on the kalman filter predictions
        for pred in next_frame_preds:
            c = h(pred)
            # cv2.circle(kalman_filter_output, (int(c[0]), int(c[1])), 1, (0, 255, 0), -1)

        full_frame = make_2_comparison(actual_color, kalman_filter_output)
        video_frames.append(full_frame)

    output_path = os.path.join("output", f"{args.name}.mp4")
    write_to_video(video_frames, output_path, True)


















        #######################################
        # Get translation prediction
        # if i > 0:
            # top_displacements = fourier_image_displacement(prev_background, background)
# 
            # # Evaluate MAE for the 3 most liekly displacements
            # mae = []
            # for d in top_displacements:
                # if np.any(np.abs(d[0]) > 64) or np.any(np.abs(d[1]) > 128):
                    # mae.append(255 * 256 * 512)
                    # continue
# 
                # prev_center_block = prev_background[64:192, 128:384]
                # new_row = [64+d[0], 192+d[0]]
                # new_col = [128+d[1], 384+d[1]]
                # center_block_shifted = background[new_row[0]:new_row[1], new_col[0]:new_col[1]]
                # mae.append(np.sum(np.abs(prev_center_block - center_block_shifted)))
# 
            # # Choose best mae
            # if np.sum(mae) < 255 * 256 * 512 * 3:
                # displacement = top_displacements[mae.index(min(mae))]
            # else:
                # displacement = np.zeros(2)
        # else:
            # displacement = np.zeros(2)
            # prev_background = background
        # print(displacement)
