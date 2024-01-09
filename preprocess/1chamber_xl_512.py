import cv2
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


# Set the desired size
resize_dim = 512
# set the video codec
fourcc_body = cv2.VideoWriter_fourcc("F", "M", "P", "4")
fourcc_ftir = cv2.VideoWriter_fourcc("F", "F", "V", "1")


def select_folder():
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select folder to process

    folder = filedialog.askdirectory(
        parent=root,
        title="Select a subfolder to process",
    )

    return folder


def process_chamber(file_path):

    # initialize directory paths
    experiment_folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    output_folder = os.path.join(experiment_folder, file_name[:-10])
    # open the video capture objects
    cap_body = cv2.VideoCapture(file_path)
    cap_ftir = cv2.VideoCapture(file_path[:-9] + "ftir.avi")
    # get the frame count and fps
    fps = int(cap_body.get(cv2.CAP_PROP_FPS))
    body_frame_count = int(cap_body.get(cv2.CAP_PROP_FRAME_COUNT))
    ftir_frame_count = int(cap_ftir.get(cv2.CAP_PROP_FRAME_COUNT))
    # make sure the frame counts are the same for both videos
    frame_count = min(body_frame_count, ftir_frame_count)

    # create a new video writer for each chamber
    body_file_name = f"{file_name[:-9]}512_body.avi"
    body_file_path = os.path.join(output_folder, body_file_name)
    body_video_writer = cv2.VideoWriter(
        body_file_path, fourcc_body, fps, (resize_dim, resize_dim)
    )
    ftir_file_name = f"{file_name[:-9]}512_ftir.avi"
    ftir_file_path = os.path.join(output_folder, ftir_file_name)
    ftir_video_writer = cv2.VideoWriter(
        ftir_file_path, fourcc_ftir, fps, (resize_dim, resize_dim)
    )

    # iterate over the frames, crop each frame into the current chamber, and save to the respective video writers
    for i in tqdm(range(frame_count)):
        ret_body, frame_body = cap_body.read()
        ret_ftir, frame_ftir = cap_ftir.read()

        if not ret_body or not ret_ftir:
            break
        # skip the first 15 frames, there is a glitch in the video at the beginning
        # this is a bug in the acquisition software, pending on a fix
        if i < 15:
            continue

        # resize the frame to the desired size
        resized_frame_body = cv2.resize(frame_body, (resize_dim, resize_dim))
        resized_frame_ftir = cv2.resize(frame_ftir, (resize_dim, resize_dim))

        # write the frame to the video writer
        body_video_writer.write(resized_frame_body)
        ftir_video_writer.write(resized_frame_ftir)

    # Release video writers and video capture objects
    body_video_writer.release()
    ftir_video_writer.release()
    cap_body.release()
    cap_ftir.release()


def main():

    # Ask the user to select folder to process
    root = tk.Tk()
    root.withdraw()

    experiment_folder = select_folder()

    if not os.path.exists(experiment_folder):
        print("The directory does not exist.")
        return

    # Iterate over all recordings in the folder

    for file_name in os.listdir(experiment_folder):
        if file_name.startswith("."):  # skip hidden files
            continue
        file_path = os.path.join(experiment_folder, file_name)
        if not file_name.lower().endswith("trans.avi"):
            continue
        print("\nstart to split recording: " + file_name[:-10])

        output_folder = os.path.join(experiment_folder, file_name[:-10])
        os.makedirs(output_folder, exist_ok=True)

        process_chamber(file_path)

    return


if __name__ == "__main__":
    main()
