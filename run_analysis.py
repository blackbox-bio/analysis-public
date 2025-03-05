from process import *
from summary import *
from dlc_runner import *
import concurrent.futures
from joblib import Parallel, delayed
import sys
import warnings

sys.path.append("./preprocess/")
from FourChamber_split_resize import *


def main():

    # Ask the user to select subfolder to process
    root = tk.Tk()
    root.withdraw()

    # input the path to the experiment folder
    experiment_folder = select_folder()

    # split and resize the 4chamber recordings
    # FourChamber_split_resize(experiment_folder, fulres=True)

    experiment_name = os.path.basename(experiment_folder)
    parent_folder = os.path.dirname(experiment_folder)
    analysis_folder = os.path.join(parent_folder, f"{experiment_name}_analysis")

    # generate the list of recordings to be processed
    recording_list = get_recording_list([analysis_folder])

    # generate the list of trans_resize.avi videos to pass to deeplabcut
    body_videos = [
        os.path.join(recording, "trans_resize.avi") for recording in recording_list
    ]

    # run deeplabcut
    # run_deeplabcut(dlc_config_path, body_videos)

    # now that done with DLC tracking, start process the recordings
    print(f"In total {len(recording_list)} videos to be processed: ")
    print(f"{[os.path.basename(recording) for recording in recording_list]}")

    num_workers = 6  # set to work with 13900K CPU
    # ignore warnings encountered during the process
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in arccos")
    warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_recording_wrapper, recording)
            for recording in recording_list
        ]



    # generate summary csv from the processed videos
    time_bins = ((0, 5),)
    generate_summary_csv(analysis_folder, time_bins)


if __name__ == "__main__":
    import tkinter as tk

    main()
