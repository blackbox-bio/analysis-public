import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import gc
import h5py
from collections import defaultdict
import cv2
from scipy.ndimage import gaussian_filter1d


def select_folder():
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select subfolder to process

    folder = filedialog.askdirectory(
        parent=root,
        title="Select a subfolder to process",
    )

    return folder


def cal_distance_(label, bodypart="tailbase"):
    """helper function for "calculate distance traveled"""
    x = gaussian_filter1d(label[bodypart]["x"].values, 3)
    y = gaussian_filter1d(label[bodypart]["y"].values, 3)
    d_x = np.diff(x)
    d_y = np.diff(y)
    d_location = np.sqrt(d_x**2 + d_y**2)
    return d_location


def four_point_transform(image, tx, ty, cx, cy, wid, length):
    """
    helper function for center and align a single video frame
    input:
        T, coord of tailbase, which is used to center the mouse
        TN, vector from tailbase to centroid
        wid, the width of the to-be-cropped portion
        length, the length of the to-be-cropped portion

    output:
        warped: the cropped portion in the size of (wid, length),
        mouse will be centered by tailbase,
        aligned by the direction from tailbase to centroid

    """
    T = np.array([tx, ty])
    N = np.array([cx, cy])
    TN = N - T

    uTN = TN / np.linalg.norm(TN)  # calculate the unit vector for TN

    # calculate the unit vector perpendicular to uTN
    uAB = np.zeros((1, 2), dtype="float32")
    uAB[0][0] = uTN[1]
    uAB[0][1] = -uTN[0]

    # calculate four corners of the to-be-cropped portion of the image
    #   use centroid to center the mouse
    A = N + uAB * (wid / 2) + uTN * (length / 2)
    B = N - uAB * (wid / 2) + uTN * (length / 2)
    C = N - uAB * (wid / 2) - uTN * (length / 2)
    D = N + uAB * (wid / 2) - uTN * (length / 2)

    # concatenate four corners into a np.array
    pts = np.concatenate((A, B, C, D))
    pts = pts.astype("float32")

    # generate the corresponding four corners in the cropped image
    dst = np.float32([[0, 0], [wid, 0], [wid, length], [0, length]])

    # generate transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)

    # rotate and crop image
    warped = cv2.warpPerspective(image, M, (wid, length))

    return warped


def denoise(luminance, noise):
    """Take a luminance signal and remove noise from it"""
    luminance = luminance - noise
    luminance[luminance < 0] = 0.0
    return luminance


def cal_paw_luminance(label, cap, size=22):
    """
    helper function for extracting the paw luminance signals of both hind paws from the ftir video

    input:
    label: DLC tracking of the recording
    ftir_video: ftir video of the recording
    size: size of the cropping window centered on a paw
    output:
    hind_left: paw luminance of the left hind paw
    hind_right: paw luminance of the right hind paw
    """

    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"video length is {num_of_frames/fps/60} mins")

    hind_right = []
    hind_left = []
    front_right = []
    front_left = []
    background_luminance = []

    # for i in tqdm(range(500)):
    for i in tqdm(range(num_of_frames)):
        frame = cap.read()[1]  # Read the next frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # calculate the luminance of the four paws
        x, y = (
            int(label["rhpaw"][["x"]].values[i]),
            int(label["rhpaw"][["y"]].values[i]),
        )
        hind_right.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["lhpaw"][["x"]].values[i]),
            int(label["lhpaw"][["y"]].values[i]),
        )
        hind_left.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["rfpaw"][["x"]].values[i]),
            int(label["rfpaw"][["y"]].values[i]),
        )
        front_right.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["lfpaw"][["x"]].values[i]),
            int(label["lfpaw"][["y"]].values[i]),
        )
        front_left.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        # calculate background luminance
        background_luminance.append(np.nanmean(frame))

    hind_right = np.array(hind_right)
    hind_left = np.array(hind_left)
    front_right = np.array(front_right)
    front_left = np.array(front_left)
    background_luminance = np.array(background_luminance)

    hind_left_mean = np.nanmean(hind_left)
    hind_right_mean = np.nanmean(hind_right)
    front_left_mean = np.nanmean(front_left)
    front_right_mean = np.nanmean(front_right)
    hind_left = np.nan_to_num(hind_left, nan=hind_left_mean)
    hind_right = np.nan_to_num(hind_right, nan=hind_right_mean)
    front_left = np.nan_to_num(front_left, nan=front_left_mean)
    front_right = np.nan_to_num(front_right, nan=front_right_mean)

    hind_left = denoise(hind_left, background_luminance)
    hind_right = denoise(hind_right, background_luminance)
    front_left = denoise(front_left, background_luminance)
    front_right = denoise(front_right, background_luminance)

    return hind_left, hind_right, front_left, front_right, background_luminance


def scale_ftir(hind_left, hind_right):
    """helper function for doing min 95-quntile scaler
    for individual recording, pool left paw and right paw ftir readings and find min and 95 percentile;
    then use those values to scale the readings"""

    left_paw = np.array(hind_left)
    right_paw = np.array(hind_right)

    min_ = min(np.nanmin(left_paw), np.nanmin(right_paw))
    max_ = max(np.nanmax(left_paw), np.nanmax(right_paw))
    quantile_ = np.nanquantile(np.concatenate([left_paw, right_paw]), 0.95)

    left_paw = (left_paw - min_) / (quantile_ - min_)
    right_paw = (right_paw - min_) / (quantile_ - min_)

    # replace all nan values with the mean, the nan values comes from DLC not tracking properly for those timepoints
    left_paw_mean = np.nanmean(left_paw)
    right_paw_mean = np.nanmean(right_paw)
    left_paw = np.nan_to_num(left_paw, nan=left_paw_mean)
    right_paw = np.nan_to_num(right_paw, nan=right_paw_mean)

    return (left_paw, right_paw)


def cal_stand_on_two_paws(front_left, front_right, threshold=0.05):
    """helper function for calculating when both of the front paws are off the ground,
    which is quantified as the average luminance of the two front paws is below a threshold.
    return a one-hot vector for when the animal is standing on two hind paws"""

    return ((front_left < threshold) * (front_right < threshold)) == 1
