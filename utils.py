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


def get_recording_list(directorys):

    recording_list = []

    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file in files:
                # file_path = os.path.join(root, file)
                if file.endswith("trans_resize.avi"):
                    recording_list.append(root)
                    # avi_files.append(os.path.join(root, file))
    return recording_list


def cal_distance_(label, bodypart="tailbase"):
    """helper function for "calculate distance traveled"""
    x = gaussian_filter1d(label[bodypart]["x"].values, 3)
    y = gaussian_filter1d(label[bodypart]["y"].values, 3)
    d_x = np.diff(x)
    d_y = np.diff(y)
    d_location = np.sqrt(d_x**2 + d_y**2)
    d_location = np.insert(d_location, 0, 0)
    return d_location


def get_distance(x1, y1, x2, y2):
    """helper function to calculate distance between two points"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def body_parts_distance(label, bp1, bp2):
    """helper function to calculate distance between two body parts"""
    x1 = label[bp1]["x"]
    y1 = label[bp1]["y"]
    x2 = label[bp2]["x"]
    y2 = label[bp2]["y"]
    return get_distance(x1, y1, x2, y2)


def get_vector(label, bp1, bp2):
    """helper function to calculate vector from bp1 to bp2"""
    x1 = label[bp1]["x"]
    y1 = label[bp1]["y"]
    x2 = label[bp2]["x"]
    y2 = label[bp2]["y"]
    return np.array([x2 - x1, y2 - y1])


def get_angle(v1, v2):
    """helper function to calculate angle between two vectors"""
    theta = np.sum(v1 * v2, axis=0) / (
        np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)
    )
    angle = np.arccos(theta) / np.pi * 180
    sign = np.sign(np.cross(v1, v2, axis=0))
    sign[sign == 0] = 1  # if cross product is 0, set sign to 1
    counterclockwise_angle = angle * sign
    return counterclockwise_angle


# def cal_body_mean_movement(label):
#     """using DLC tracking of several main body parts to calculate mean body movement
#        first calculate the frame to frame speed for each body part, then average them
#        return body_mean_movement"""
#     bodyparts = ['tailbase', 'centroid', 'neck', 'snout', 'hlpaw', 'hrpaw', 'flpaw', 'frpaw']
#     place_holder = {}
#     for body in bodyparts:
#         place_holder[body] = cal_distance_(label, body)
#
#     return np.mean(np.vstack([place_holder[body] for body in bodyparts]).T, axis=1, keepdims=True)


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

    # num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # print(f"video length is {num_of_frames/fps/60} mins")

    hind_right = []
    hind_left = []
    front_right = []
    front_left = []
    background_luminance = []

    # loop infinitely because we cannot trust `CAP_PROP_FRAME_COUNT`
    # https://stackoverflow.com/questions/31472155/python-opencv-cv2-cv-cv-cap-prop-frame-count-get-wrong-numbers
    # for i in tqdm(range(500)):
    i = 0
    while True:
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break

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

        i += 1

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

    return hind_left, hind_right, front_left, front_right, background_luminance, i


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


def both_front_paws_lifted(front_left, front_right, threshold=1e-4):
    """helper function for calculating when both of the front paws are off the ground,
    which is quantified as the average luminance of the two front paws is below a threshold.
    return a one-hot vector for when the animal is standing on two hind paws"""

    return ((front_left < threshold) * (front_right < threshold)) == 1
