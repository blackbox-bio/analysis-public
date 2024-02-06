import cv2
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


# Set the desired size
full_dim = 1024
# set the video codec
fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
chambers = [f"chamber_{i}" for i in range(1, 5)]
coords = {
    "chamber_1": [(0, 0), (1024, 1024)],
    "chamber_2": [(1024, 0), (2048, 1024)],
    "chamber_3": [(0, 1024), (1024, 2048)],
    "chamber_4": [(1024, 1024), (2048, 2048)],
}


def select_folder():
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select folder to process

    folder = filedialog.askdirectory(
        parent=root,
        title="Select a subfolder to process",
    )

    return folder


def process_chamber(file_path, chamber):

    # initialize directory paths
    recording_folder = os.path.dirname(file_path)
    parent_folder = os.path.dirname(recording_folder)
    analysis_folder = os.path.join(parent_folder, f"{recording_folder}_analysis")
    file_name = os.path.basename(file_path)
    output_folder = os.path.join(analysis_folder, file_name[:-10] + chamber)
    os.makedirs(output_folder, exist_ok=True)
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
    body_file_name = "trans_raw.avi"
    body_file_path = os.path.join(output_folder, body_file_name)
    body_video_writer = cv2.VideoWriter(
        body_file_path, fourcc, fps, (full_dim, full_dim)
    )
    ftir_file_name = "ftir_raw.avi"
    ftir_file_path = os.path.join(output_folder, ftir_file_name)
    ftir_video_writer = cv2.VideoWriter(
        ftir_file_path, fourcc, fps, (full_dim, full_dim)
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

        # read the coordinates for cropping the frame
        x1, y1 = coords[chamber][0]
        x2, y2 = coords[chamber][1]

        # crop the frame and pad it to the desired size
        smaller_frame_body = frame_body[y1:y2, x1:x2]
        smaller_frame_ftir = frame_ftir[y1:y2, x1:x2]

        # pad the frame to the desired size
        pad_x = max(0, (full_dim - smaller_frame_body.shape[0]))
        pad_y = max(0, (full_dim - smaller_frame_body.shape[1]))
        resized_frame_body = np.pad(
            smaller_frame_body, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant"
        )
        resized_frame_ftir = np.pad(
            smaller_frame_ftir, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant"
        )

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
    # Using multithreading to speed up the process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for file_name in os.listdir(experiment_folder):
            file_path = os.path.join(experiment_folder, file_name)
            if not file_name.lower().endswith("trans.avi"):
                continue
            if file_name.startswith("."):  # skip hidden files
                continue
            print("\nstart to split recording: " + file_name[:-10])

            futures = [
                executor.submit(process_chamber, file_path, chamber)
                for chamber in chambers
            ]
            concurrent.futures.wait(futures)

    return


if __name__ == "__main__":
    main()
