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
from scipy.ndimage import label as label_connected_components
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
                if file == "trans.mp4":
                    recording_list.append(root)
    return recording_list


def cal_centroid_distance_delta(label):
    """
    Per-frame distance moved by the body centroid, used for "distance traveled".

    Note: this is not interchangeable with get_speed. This tracks the
    likelihood-weighted centroid (see cal_centroid) and smooths the positions
    before taking the frame-to-frame distance, whereas get_speed tracks a
    single body part and smooths the speed after taking the distance.

    Parameters
    ----------
    label : pd.DataFrame
        DLC tracking DataFrame with MultiIndex columns: (bodypart, coord).

    Returns
    -------
    d_location : np.ndarray
        Distance (in pixels) moved by the centroid since the previous frame.
        The first frame is 0. NaN where the centroid is undefined.
    """

    centroid = cal_centroid(label)
    x = gaussian_filter1d(centroid[:, 0], 3)
    y = gaussian_filter1d(centroid[:, 1], 3)
    d_x = np.diff(x)
    d_y = np.diff(y)
    d_location = np.sqrt(d_x**2 + d_y**2)
    d_location = np.insert(d_location, 0, 0)
    return d_location

def cal_centroid(label):
    """
    input:
    label: DLC tracking of the recording
    return: the x,y location of the (estimated) centroid of the mouse
    """

    # for now hard-code the list of bodyparts used to estimate centroid
    bp_list = [
        # 'tailbase',
        'hip',
        'sternumtail',
        'sternumhead',
        'neck',
        # 'snout',
        'lhip',
        'rhip',
        'lshoulder',
        'rshoulder'
    ]

    sub = label.loc[:, pd.IndexSlice[bp_list, ['x', 'y', 'likelihood']]]
    x = sub.xs('x', axis=1, level=-1)
    y = sub.xs('y', axis=1, level=-1)
    lik = sub.xs('likelihood', axis=1, level=-1)

    # likelihood-weighted sum that integrate different bodyparts locations
    weighted_x = (x * lik).sum(axis=1)
    weighted_y = (y * lik).sum(axis=1)
    sum_weights = lik.sum(axis=1)
    mean_x = weighted_x.div(sum_weights).where(sum_weights.ne(0))
    mean_y = weighted_y.div(sum_weights).where(sum_weights.ne(0))

    centroid = pd.concat([mean_x, mean_y], axis=1)
    centroid.columns = ['x', 'y']
    centroid = centroid.to_numpy()

    return centroid

def cal_displacement(
        centroid_x,
        centroid_y,
        fps,
        window_sec: float = 0.5,
        smooth_sigma: int = 3
):
    """
    Calculate the displacement of the animal relative to a rolling mean location
    in the past N seconds, and save it to the same features.h5 file.

    Parameters
    ----------
    centroid: centroid tracking time series
    fps: fps of the recording
    window_sec : float, optional
        Size of rolling window in seconds (default = 0.5).
    smooth_sigma : int, optional
        Gaussian smoothing sigma in frames to reduce jitter (default = 3).

    Returns
    -------
    displacement_px : np.ndarray
        Displacement (in pixels) from rolling mean location.
    """

    # --- basic setup ---
    half_window = int((window_sec * fps) / 2)

    # --- smooth position ---
    x_smooth = gaussian_filter1d(centroid_x, sigma=smooth_sigma)
    y_smooth = gaussian_filter1d(centroid_y, sigma=smooth_sigma)

    n = len(x_smooth)
    displacement_px = np.zeros(n)

    # --- compute centered displacement ---
    for i in range(n):
        left = max(i - half_window, 0)
        right = min(i + half_window, n - 1)
        dx = x_smooth[right] - x_smooth[left]
        dy = y_smooth[right] - y_smooth[left]
        displacement_px[i] = np.hypot(dx, dy)

    return displacement_px


# duplicate from Ethos start ---------------------------

def label_locomotion(displacement_px, fps, reference_threshold=20, reference_fps=45, duration_s=1.0):
    """
    Label locomotion frames based on per-frame displacement, with threshold scaled by fps.
    A frame is locomotion if its displacement exceeds the threshold and it belongs to a
    run of such frames lasting at least duration_s.

    Parameters
    ----------
    displacement_px : np.ndarray or pd.Series
        Per-frame displacement in pixels from the rolling mean centroid location (output of cal_displacement).
    fps : float
        Frame rate of the recording.
    reference_threshold : float
        Displacement threshold in pixels for locomotion at the reference_fps (default: 20 px at 45 fps).
    reference_fps : float
        The FPS at which the reference_threshold is defined (default: 45).
    duration_s : float
        Minimum duration in seconds of a continuous above-threshold run to count as locomotion (default: 1.0).

    Returns
    -------
    locomotion_mask : np.ndarray of bool
        Boolean array where True indicates locomotion.
    """
    # Adjust the threshold proportionally to frame rate
    scaled_threshold = reference_threshold * (fps / reference_fps)
    locomotion_mask = displacement_px > scaled_threshold

    # Enforce minimum duration
    min_duration = int(duration_s * fps)
    labeled, n = label_connected_components(locomotion_mask)
    locomotion_long = np.zeros_like(locomotion_mask)

    for i in range(1, n + 1):
        idx = np.where(labeled == i)[0]
        if len(idx) >= min_duration:
            locomotion_long[idx] = 1

    return locomotion_long.astype(bool)

def label_not_moving(label, fps, reference_px_per_frame=0.5, reference_fps=45, duration_s=0.5):
    """
    Label frames as 'not moving' based on per-frame speed threshold scaled with fps.

    Parameters
    ----------
    label : pandas.DataFrame
        DLC tracking DataFrame.
    fps : float
        Frame rate of the video.
    reference_px_per_frame : float
        Pixel/frame threshold at reference_fps (default = 0.5 at 45 fps).
    reference_fps : float
        The base FPS for which reference_px_per_frame is defined.
    duration_s : float
        Minimum duration (in seconds) of continuous stillness (default = 0.5).

    Returns
    -------
    still_long : np.ndarray
        Boolean array marking long stillness segments (True = still).
    """
    # List of body parts to evaluate for stillness.
    bp_list = [
        "hip",
        "sternumtail",
        "sternumhead",
        "neck",
        "snout",
        "lhpaw",
        "rhpaw",
        "lfpaw",
        "rfpaw",
    ]

    # Scale threshold based on fps (inverse relation)
    px_per_frame_thresh = reference_px_per_frame * (reference_fps / fps)

    # Compute smoothed speeds for all body parts
    speeds = [get_speed(label, bp) for bp in bp_list]
    speeds = np.vstack(speeds)

    # Frame is still if all body parts are below threshold
    still_mask = np.all(speeds < px_per_frame_thresh, axis=0)

    # Enforce minimum duration
    min_duration = int(duration_s * fps)
    labeled, n = label_connected_components(still_mask)
    still_long = np.zeros_like(still_mask)

    for i in range(1, n + 1):
        idx = np.where(labeled == i)[0]
        if len(idx) >= min_duration:
            still_long[idx] = 1

    return still_long.astype(bool)


def get_speed(label, bp, filter_size = 3):
    """
    helper function to calculate the speed (frame-to-frame delta distance) of a body part
    """

    label = filter_tracking_by_likelihood(label)

    x = label[bp]["x"].copy()
    y = label[bp]["y"].copy()

    # Compute frame-to-frame displacement
    d_x = np.diff(x, prepend=x.iloc[0])
    d_y = np.diff(y, prepend=y.iloc[0])
    speed = np.sqrt(d_x ** 2 + d_y ** 2)

    # Apply Gaussian smoothing
    smoothed_speed = gaussian_filter1d(speed, sigma=filter_size)

    return smoothed_speed


def get_angular_velocity(label: pd.DataFrame, bp1: str, bp2: str, filter_size: int = 3) -> np.ndarray:
    """
    Helper function to calculate smoothed angular velocity (in degrees/frame)
    between two body parts across frames. The vector is defined as bp2 -> bp1

    Parameters
    ----------
    label : pd.DataFrame
        DLC tracking DataFrame with MultiIndex columns: (bodypart, coord).
    bp1 : str
        Name of the front body part (e.g. 'snout').
    bp2 : str
        Name of the back body part (e.g. 'sternumtail').
    filter_size : int
        Gaussian smoothing filter sigma.

    Returns
    -------
    smoothed_angular_velocity : np.ndarray
        Smoothed angular velocity in degrees per frame.
    """
    label = filter_tracking_by_likelihood(label)

    x1 = label[bp1]["x"].copy()
    y1 = label[bp1]["y"].copy()
    x2 = label[bp2]["x"].copy()
    y2 = label[bp2]["y"].copy()

    # Compute orientation angle per frame
    theta = np.arctan2(y1 - y2, x1 - x2)
    theta_unwrapped = np.unwrap(theta)

    # Compute angular velocity in degrees
    d_theta = np.diff(theta_unwrapped, prepend=theta_unwrapped[0])
    angular_velocity_deg = np.degrees(d_theta)

    # Apply Gaussian smoothing
    smoothed_angular_velocity = gaussian_filter1d(angular_velocity_deg, sigma=filter_size)

    return smoothed_angular_velocity

def label_turning(
    label,
    fps,
    threshold_deg_per_s=45,
    duration_s=0.4,
    smooth_sigma=3,
    bp1="snout",
    bp2="tailbase"
):
    """
    Label turning behavior based on angular velocity between two body parts.
    Uses get_angular_velocity() to compute smoothed angular velocity.

    Parameters
    ----------
    label : pd.DataFrame
        DLC tracking DataFrame.
    fps : float
        Frame rate of video.
    threshold_deg_per_s : float
        Angular velocity threshold in deg/sec.
    duration_s : float
        Minimum turning duration in seconds to count as a turn.
    smooth_sigma : float
        Smoothing applied inside get_angular_velocity (in frames).
    bp1 : str
        Front body part (e.g., "snout").
    bp2 : str
        Rear body part (e.g., "tailbase").

    Returns
    -------
    turning_labels : np.ndarray
        Array of same length as frames:
        - 0 = not turning
        - 2 = left turn (ang_vel < -threshold)
        - 3 = right turn (ang_vel > threshold)
    """
    # Get angular velocity in deg/frame
    ang_vel = get_angular_velocity(label, bp1=bp1, bp2=bp2, filter_size=smooth_sigma)

    # Convert threshold to deg/frame
    threshold = threshold_deg_per_s / fps
    min_duration = int(duration_s * fps)

    # Init label array
    turning_labels = np.zeros_like(ang_vel, dtype=int)

    # Label left turns
    left_mask = ang_vel > threshold
    labeled_left, n_left = label_connected_components(left_mask)
    for i in range(1, n_left + 1):
        idx = np.where(labeled_left == i)[0]
        if len(idx) >= min_duration:
            turning_labels[idx] = 2

    # Label right turns
    right_mask = ang_vel < -threshold
    labeled_right, n_right = label_connected_components(right_mask)
    for i in range(1, n_right + 1):
        idx = np.where(labeled_right == i)[0]
        if len(idx) >= min_duration:
            turning_labels[idx] = 3

    return turning_labels

# duplicate from Ethos end ---------------------------


def filter_tracking_by_likelihood(label: pd.DataFrame, likelihood_thresh: float = 0.6) -> pd.DataFrame:
    """
    Filter all body parts in a DLC tracking DataFrame by likelihood.

    Low-confidence x/y values (likelihood < threshold) are replaced by NaN and then filled.

    Parameters
    ----------
    label : pd.DataFrame
        DLC tracking DataFrame with MultiIndex columns: (bodypart, coord), e.g., ('snout', 'x')
    likelihood_thresh : float
        Minimum confidence value required to retain a tracking point (default: 0.6)

    Returns
    -------
    filtered_label : pd.DataFrame
        Modified DataFrame with low-confidence positions removed and interpolated
    """
    filtered_label = label.copy()

    # Loop through all body parts
    for bp in label.columns.levels[0]:
        if (bp, 'likelihood') not in label.columns:
            continue  # skip untracked parts

        x = filtered_label[(bp, "x")]
        y = filtered_label[(bp, "y")]
        likelihood = filtered_label[(bp, "likelihood")]

        # Mask low-confidence values
        low_confidence = likelihood < likelihood_thresh
        x[low_confidence] = pd.NA
        y[low_confidence] = pd.NA

        # Fill gaps with backward then forward fill
        filtered_label[(bp, "x")] = x.bfill().ffill()
        filtered_label[(bp, "y")] = y.bfill().ffill()

    return filtered_label


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
    # z component of the 2D cross product. written out because numpy 2 no longer
    # supports np.cross on 2D vectors
    sign = np.sign(v1[0] * v2[1] - v1[1] * v2[0])
    sign[sign == 0] = 1  # if cross product is 0, set sign to 1
    counterclockwise_angle = angle * sign
    return counterclockwise_angle


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

    # extract the paw coordinates once. indexing the DataFrame inside the frame
    # loop copies the whole column every frame, which makes the loop O(n^2)
    paw_xy = {paw: label[paw][["x", "y"]].to_numpy() for paw in paws}
    DLC_tracking_length = len(label)

    # the loop never runs past the end of the tracking, so preallocate arrays
    # of that length and trim them to the number of frames read afterwards.
    # appending to lists of numpy scalars uses ~5x the memory.
    # dtypes match what np.array() produced from the old lists
    paw_luminescence = {paw: np.empty(DLC_tracking_length, np.uint64) for paw in paws}
    paw_luminance = {paw: np.empty(DLC_tracking_length, np.float64) for paw in paws}
    paw_print_size = {paw: np.empty(DLC_tracking_length, np.int64) for paw in paws}

    background_luminance = np.empty(DLC_tracking_length, np.float64)

    # legacy paw luminance calculation
    hind_right = np.empty(DLC_tracking_length, np.float64)
    hind_left = np.empty(DLC_tracking_length, np.float64)
    front_right = np.empty(DLC_tracking_length, np.float64)
    front_left = np.empty(DLC_tracking_length, np.float64)
    # legacy end----------------

    expected_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"expected_total: {expected_total}, DLC_tracking_length: {DLC_tracking_length}")

    # expected_total = min(expected_total, DLC_tracking_length) # take the minimum of the two

    i = 0
    pbar = tqdm(
        total=expected_total, dynamic_ncols=True, desc="paw luminance calculation"
    )

    while True:
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break

        # workaround: if the ftir video is longer than DLC tracking, exit to
        # avoid index error
        if DLC_tracking_length <= i:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # calculate background luminance
        background_luminance[i] = np.mean(frame)

        # legacy paw luminance calculation
        x, y = int(paw_xy["rhpaw"][i, 0]), int(paw_xy["rhpaw"][i, 1])
        hind_right[i] = np.nanmean(frame[y - size : y + size, x - size : x + size])

        x, y = int(paw_xy["lhpaw"][i, 0]), int(paw_xy["lhpaw"][i, 1])
        hind_left[i] = np.nanmean(frame[y - size : y + size, x - size : x + size])

        x, y = int(paw_xy["rfpaw"][i, 0]), int(paw_xy["rfpaw"][i, 1])
        front_right[i] = np.nanmean(frame[y - size : y + size, x - size : x + size])

        x, y = int(paw_xy["lfpaw"][i, 0]), int(paw_xy["lfpaw"][i, 1])
        front_left[i] = np.nanmean(frame[y - size : y + size, x - size : x + size])
        # legacy paw luminance calculation end----------------

        frame_denoise, paw_print = get_ftir_mask(frame)

        # calculate the luminance of the four paws
        for paw in paws:
            x, y = int(paw_xy[paw][i, 0]), int(paw_xy[paw][i, 1])
            luminescence, print_size, luminance = get_individual_paw_luminance(
                frame_denoise, paw_print, x, y, size
            )
            paw_luminescence[paw][i] = luminescence
            paw_print_size[paw][i] = print_size
            paw_luminance[paw][i] = luminance

        i += 1

        pbar.update(1)

    pbar.close()

    # trim the preallocated arrays to the number of frames actually read
    background_luminance = background_luminance[:i]
    for dict_ in [paw_luminescence, paw_print_size, paw_luminance]:
        for paw in paws:
            dict_[paw] = dict_[paw][:i]
            mean = np.nanmean(dict_[paw])
            dict_[paw] = np.nan_to_num(dict_[paw], nan=mean)

    for paw in paws:
        paw_luminance[paw] = denoise(paw_luminance[paw], background_luminance)

    # legacy paw luminance calculation
    hind_right = hind_right[:i]
    hind_left = hind_left[:i]
    front_right = front_right[:i]
    front_left = front_left[:i]
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
