import cv2
import numpy as np

from utils import *


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


def cal_paw_luminance_rework(label, cap, size=22):

    paws = ["lhpaw", "rhpaw", "lfpaw", "rfpaw"]

    paw_luminescence = {paw: [] for paw in paws}
    paw_luminance = {paw: [] for paw in paws}
    paw_print_size = {paw: [] for paw in paws}

    background_luminance = []

    i = 0
    while True:
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # calculate background luminance
        background_luminance.append(np.mean(frame))

        frame_denoise, paw_print = get_ftir_mask(frame)

        # calculate the luminance of the four paws
        for paw in paws:
            x, y = (
                int(label[paw][["x"]].values[i]),
                int(label[paw][["y"]].values[i]),
            )
            luminescence, print_size, luminance = get_individual_paw_luminance(
                frame_denoise, paw_print, x, y, size
            )
            paw_luminescence[paw].append(luminescence)
            paw_print_size[paw].append(print_size)
            paw_luminance[paw].append(luminance)

        i += 1
    background_luminance = np.array(background_luminance)
    for dict_ in [paw_luminescence, paw_print_size, paw_luminance]:
        for paw in paws:
            dict_[paw] = np.array(dict_[paw])
            mean = np.nanmean(dict_[paw])
            dict_[paw] = np.nan_to_num(dict_[paw], nan=mean)

    for paw in paws:
        paw_luminance[paw] = denoise(paw_luminance[paw], background_luminance)

    return paw_luminescence, paw_print_size, paw_luminance, background_luminance, i
