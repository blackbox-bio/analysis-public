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

import tkinter as tk
from tkinter import filedialog

from process import *
from summary import *
from dlc_runner import *


def main():

    # Ask the user to select subfolder to process
    root = tk.Tk()
    root.withdraw()

    # input the path to the experiment folder
    exp_folder = select_folder()

    # run dlc
    body_videos = get_body_videos([exp_folder])
    run_deeplabcut(dlc_config_path, body_videos)


    features_folder = os.path.join(exp_folder, "features")
    if not os.path.exists(features_folder):
        # Create the directory
        os.makedirs(features_folder)



    # generate the list of videos to be processed
    video_list = []
    for i in os.listdir(os.path.join(exp_folder, "videos")):
        if "body" in i and ".avi" in i:
            video_list.append(i.split("_body")[0])

    print(f"In total {len(video_list)} videos to be processed: \n{video_list}")

    # Process the videos iteratively
    for video in video_list:
        process_video(video, exp_folder, features_folder)

    # generate summary csv from the processed videos
    summary_csv = os.path.join(exp_folder, "summary.csv")
    generate_summary_csv(features_folder, summary_csv)


if __name__ == "__main__":
    main()
