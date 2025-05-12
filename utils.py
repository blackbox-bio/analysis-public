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
from scipy.ndimage import median_filter
from dataclasses import dataclass
from typing import Dict
from palmreader_analysis.variants import LuminanceMeasure, Paw


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


def detect_animal_in_recording(label, fps, likelihood_threshold=0.5, temp_threshood=10):
    """
    :param label: DLC tracking file
    :param fps: fps of the recording
    :param likelihood_threshold
    :param temp_threshood: use 10 seconds as the threshold for the mouse to be considered as detected
    :return: a boolean mask for whether the mouse is detected in the recording for each frame
    """
    likelihood_columns = [col for col in label.columns if "likelihood" in col]
    likelihood = label[likelihood_columns].values
    likelihood = np.mean(likelihood, axis=1)
    detection = likelihood > likelihood_threshold
    detection = median_filter(detection, size=temp_threshood * fps + 1)

    return detection


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
    pbar = tqdm(total=None, dynamic_ncols=True, desc="legacy paw luminance calculation")
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

        pbar.update(1)

    pbar.close()

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


# paw luminance rework ------
def get_ftir_mask(ftir_frame_gray):
    """
    Get the paw print mask from the FTIR frame. The FTIR frame is first denoised by removing the background noise.
    The paw print mask is then obtained by applying a threshold to the denoised FTIR frame.

    return the denoised FTIR frame and the paw print mask.
    """
    background_threshold = (
        17  # hard-coded the threshold, the same threshold used for the ftir heatmap
    )
    paw_print_threshold = 10

    # blur ftir frame
    ftir_frame_gray_blur = cv2.GaussianBlur(ftir_frame_gray, (3, 3), 0)

    # remove background noise
    mask = ftir_frame_gray_blur > background_threshold
    mask = mask.astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=3)
    # apply the mask to the blurred ftir frame
    ftir_frame_gray_denoise = ftir_frame_gray_blur.copy()
    mask = mask.astype(bool)
    ftir_frame_gray_denoise[~mask] = 0

    # apply the paw print threshold to get the paw print mask, boolean
    paw_print = ftir_frame_gray_denoise > paw_print_threshold
    # get the denoised ftir frame
    ftir_frame_final = ftir_frame_gray.copy()
    ftir_frame_final[~paw_print] = 0

    # get the paw_print as a frame
    ftir_mask = paw_print.astype(np.uint8) * 255

    return ftir_frame_final, ftir_mask


def get_individual_paw_luminance(ftir_frame, ftir_mask, x, y, size=22):
    """
    Get the paw luminescence, paw print size, and paw luminance.
    paw luminescence is the sum of the pixel values in the paw print mask.
    paw print size is the number of pixels in the paw print mask.
    paw luminance is the paw luminescence divided by the paw print size.
    :param ftir_frame: denoised FTIR frame
    :param ftir_mask: ftir mask
    :param x: x coordinate of the paw
    :param y: y coordinate of the paw
    :param size: size of the square region around the paw to calculate the paw luminance
    :return: paw luminescence, paw print size, paw luminance
    """
    paw_luminescence = np.nansum(ftir_frame[y - size : y + size, x - size : x + size])
    ftir_mask = ftir_mask.astype(bool)
    paw_print_size = np.sum(ftir_mask[y - size : y + size, x - size : x + size])
    paw_luminance = paw_luminescence / paw_print_size if paw_print_size > 0 else 0.0

    return paw_luminescence, paw_print_size, paw_luminance


@dataclass
class LegacyPawLuminanceData:
    """Data class for legacy paw luminance data"""

    hind_left: np.ndarray
    hind_right: np.ndarray
    front_left: np.ndarray
    front_right: np.ndarray

    def get_paw(self, paw: Paw) -> np.ndarray:
        if paw == Paw.LEFT_HIND:
            return self.hind_left
        elif paw == Paw.RIGHT_HIND:
            return self.hind_right
        elif paw == Paw.LEFT_FRONT:
            return self.front_left
        elif paw == Paw.RIGHT_FRONT:
            return self.front_right
        else:
            raise ValueError(f"Invalid paw: {paw}")


@dataclass
class PawLuminanceData:
    """Data class for paw luminance data"""

    paw_luminescence: Dict[str, list]
    paw_print_size: Dict[str, list]
    paw_luminance: Dict[str, list]
    background_luminance: np.ndarray
    frame_count: int
    legacy_paw_luminance: LegacyPawLuminanceData

    def get_measure(self, measure: LuminanceMeasure) -> Dict[str, list]:
        if measure == LuminanceMeasure.LUMINANCE:
            return self.paw_luminance
        elif measure == LuminanceMeasure.LUMINESCENCE:
            return self.paw_luminescence
        elif measure == LuminanceMeasure.PRINT_SIZE:
            return self.paw_print_size
        else:
            raise ValueError(f"Invalid measure: {measure}")


def cal_paw_luminance_rework(label, cap, size=22):

    # # debug
    # print("calling cal_paw_luminance_rework")

    paws = ["lhpaw", "rhpaw", "lfpaw", "rfpaw"]

    paw_luminescence = {paw: [] for paw in paws}
    paw_luminance = {paw: [] for paw in paws}
    paw_print_size = {paw: [] for paw in paws}

    background_luminance = []

    # legacy paw luminance calculation
    hind_right = []
    hind_left = []
    front_right = []
    front_left = []
    # legacy end----------------

    expected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    DLC_tracking_length = label["snout"][["x"]].shape[0]

    expected_total = min(expected_total, DLC_tracking_length) # take the minimum of the two

    i = 0
    pbar = tqdm(
        total=expected_total, dynamic_ncols=True, desc="paw luminance calculation"
    )

    while 1<expected_total:
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break

        # workaround: if the ftir video is longer than DLC tracking, exit to
        # avoid index error
        if len(label["rhpaw"]) <= i:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # calculate background luminance
        background_luminance.append(np.mean(frame))

        # legacy paw luminance calculation
        x, y = (
            int(label["rhpaw"][["x"]].values[i].item()),
            int(label["rhpaw"][["y"]].values[i].item()),
        )
        hind_right.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["lhpaw"][["x"]].values[i].item()),
            int(label["lhpaw"][["y"]].values[i].item()),
        )
        hind_left.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["rfpaw"][["x"]].values[i].item()),
            int(label["rfpaw"][["y"]].values[i].item()),
        )
        front_right.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))

        x, y = (
            int(label["lfpaw"][["x"]].values[i].item()),
            int(label["lfpaw"][["y"]].values[i].item()),
        )
        front_left.append(np.nanmean(frame[y - size : y + size, x - size : x + size]))
        # legacy paw luminance calculation end----------------

        frame_denoise, paw_print = get_ftir_mask(frame)

        # calculate the luminance of the four paws
        for paw in paws:
            x, y = (
                int(label[paw][["x"]].values[i].item()),
                int(label[paw][["y"]].values[i].item()),
            )
            luminescence, print_size, luminance = get_individual_paw_luminance(
                frame_denoise, paw_print, x, y, size
            )
            paw_luminescence[paw].append(luminescence)
            paw_print_size[paw].append(print_size)
            paw_luminance[paw].append(luminance)

        i += 1

        pbar.update(1)

    pbar.close()

    background_luminance = np.array(background_luminance)
    for dict_ in [paw_luminescence, paw_print_size, paw_luminance]:
        for paw in paws:
            dict_[paw] = np.array(dict_[paw])
            mean = np.nanmean(dict_[paw])
            dict_[paw] = np.nan_to_num(dict_[paw], nan=mean)

    for paw in paws:
        paw_luminance[paw] = denoise(paw_luminance[paw], background_luminance)

    # legacy paw luminance calculation
    hind_right = np.array(hind_right)
    hind_left = np.array(hind_left)
    front_right = np.array(front_right)
    front_left = np.array(front_left)
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

    legacy_paw_luminance = LegacyPawLuminanceData(
        hind_left=hind_left,
        hind_right=hind_right,
        front_left=front_left,
        front_right=front_right,
    )
    # legacy paw luminance calculation end----------------

    return PawLuminanceData(
        paw_luminescence=paw_luminescence,
        paw_print_size=paw_print_size,
        paw_luminance=paw_luminance,
        background_luminance=background_luminance,
        frame_count=i,
        legacy_paw_luminance=legacy_paw_luminance,
    )


def cal_orientation_vector(label, alpha=2.0, beta=1.0):
    """
    helper function for calculating the orientation vector of the mouse
    input:
        label: DLC tracking of the recording
        alpha: primary vector weight
        beta: secondary vector weight
    output:
        final_orientation_vector_normalized: the normalized orientation vector of the mouse
    """

    body_parts = [
        "snout",
        "neck",
        "sternumhead",
        "sternumtail",
        "hip",
        "tailbase",
    ]

    # primary orientation vector: tailbase to snout
    # secondary orientation vectors: every segment to the next segment

    # primary orientation vector
    primary_vector = (
        label["snout"][["x", "y"]].values - label["tailbase"][["x", "y"]].values
    )
    # add the average likelihood of the two points
    primary_likelihood = (
        label["snout"]["likelihood"].values + label["tailbase"]["likelihood"].values
    ) / 2

    secondary_vectors = []
    secondary_likelihoods = []
    for i in range(len(body_parts) - 1):
        secondary_vector = (
            label[body_parts[i + 1]][["x", "y"]].values
            - label[body_parts[i]][["x", "y"]].values
        )
        secondary_vectors.append(secondary_vector)
        secondary_likelihood = (
            label[body_parts[i + 1]]["likelihood"].values
            + label[body_parts[i]]["likelihood"].values
        ) / 2
        secondary_likelihoods.append(secondary_likelihood)
    # make secondary vectors into a numpy array
    secondary_vectors = np.array(secondary_vectors)
    secondary_likelihoods = np.array(secondary_likelihoods)
    secondary_likelihoods = secondary_likelihoods[:, :, np.newaxis]

    # weight the primary vector
    weighted_primary_vector = alpha * primary_vector * primary_likelihood[:, np.newaxis]

    # weight the secondary vectors
    weighted_secondary_vectors = (
        beta * secondary_vectors * secondary_likelihoods / secondary_vectors.shape[0]
    )

    # sum the weighted vectors
    weighted_secondary_sum = np.sum(weighted_secondary_vectors, axis=0)
    final_orientation_vector = weighted_primary_vector + weighted_secondary_sum

    # normalize the final orientation vector

    # check for rows where all values are zero
    is_zero_row = np.all(final_orientation_vector == 0, axis=1)
    # get the last non-zero index
    last_non_zero_index = np.where(~is_zero_row)[0][-1]
    # slice the final orientation vector to the last non-zero index
    final_orientation_vector_trimmed = final_orientation_vector[
        : last_non_zero_index + 1
    ]
    # normalize the final orientation vector
    norms_trimmed = np.linalg.norm(
        final_orientation_vector_trimmed, axis=1, keepdims=True
    )
    # avoid division by zero
    norms_trimmed[norms_trimmed == 0] = 1e-10
    final_orientation_vector_normalized = (
        final_orientation_vector_trimmed / norms_trimmed
    )

    return final_orientation_vector_normalized


# def four_point_transform(image, tx, ty, cx, cy, wid, length):
#     """
#     helper function for center and align a single video frame
#     input:
#         T, coord of tailbase, which is used to center the mouse
#         TN, vector from tailbase to centroid
#         wid, the width of the to-be-cropped portion
#         length, the length of the to-be-cropped portion
#
#     output:
#         warped: the cropped portion in the size of (wid, length),
#         mouse will be centered by tailbase,
#         aligned by the direction from tailbase to centroid
#
#     """
#     T = np.array([tx, ty])
#     N = np.array([cx, cy])
#     TN = N - T
#
#     uTN = TN / np.linalg.norm(TN)  # calculate the unit vector for TN
#
#     # calculate the unit vector perpendicular to uTN
#     uAB = np.zeros((1, 2), dtype="float32")
#     uAB[0][0] = uTN[1]
#     uAB[0][1] = -uTN[0]
#
#     # calculate four corners of the to-be-cropped portion of the image
#     #   use centroid to center the mouse
#     A = N + uAB * (wid / 2) + uTN * (length / 2)
#     B = N - uAB * (wid / 2) + uTN * (length / 2)
#     C = N - uAB * (wid / 2) - uTN * (length / 2)
#     D = N + uAB * (wid / 2) - uTN * (length / 2)
#
#     # concatenate four corners into a np.array
#     pts = np.concatenate((A, B, C, D))
#     pts = pts.astype("float32")
#
#     # generate the corresponding four corners in the cropped image
#     dst = np.float32([[0, 0], [wid, 0], [wid, length], [0, length]])
#
#     # generate transform matrix
#     M = cv2.getPerspectiveTransform(pts, dst)
#
#     # rotate and crop image
#     warped = cv2.warpPerspective(image, M, (wid, length))
#
#     return warped


def four_point_transform(frame, orientation_frame, center, width, height):
    """
    :param frame: a single frame
    :param orientation_frame: the orientation of the animal in the given frame
    :param center: the center of the animal in the given frame
    :param width: the width of the transformed frame
    :param height: the height of the transformed frame
    :return: the transformed frame in the size of (width, height), with the animal centered and aligned
    """

    orientation_vector = np.array(orientation_frame)
    center = np.array(center)

    # calculate the unit vector perpendicular to the orientation vector
    perpendicular_vector = np.array([orientation_vector[1], -orientation_vector[0]])

    # calculate the four corners of the transformed frame
    A = center + (width / 2) * perpendicular_vector + (height / 2) * orientation_vector
    B = center - (width / 2) * perpendicular_vector + (height / 2) * orientation_vector
    C = center - (width / 2) * perpendicular_vector - (height / 2) * orientation_vector
    D = center + (width / 2) * perpendicular_vector - (height / 2) * orientation_vector

    # concatenate four corners into a single array
    pts = np.array([A, B, C, D], dtype="float32")

    # generate the corresponding four corners of the output frame
    output_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )

    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, output_pts)

    # apply the perspective transform
    transformed_frame = cv2.warpPerspective(frame, M, (width, height))

    return transformed_frame
