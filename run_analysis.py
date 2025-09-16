import palmreader_analysis
from process import *
from summary import *
from dlc_runner import *
import concurrent.futures
from joblib import Parallel, delayed
import sys
import warnings
import os
import argparse

sys.path.append("./preprocess/")


def parse_time_bins(bin_str):
    """
    Parse a string like "0,-1 3,5 3,-1" into ((0, -1), (3, 5), (3, -1))
    """
    bins = []
    for part in bin_str.split():
        start, end = part.split(",")
        bins.append((int(start), int(end)))
    return tuple(bins)

def main():
    parser = argparse.ArgumentParser(
        description="Run palmreader analysis on experiment folder"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Path to the experiment folder",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of worker processes (default: 6, good for 13900K CPU)",
    )
    parser.add_argument(
        "--time_bins",
        type=str,
        default="0,-1",
        help=(
            "Space-separated list of time bins as start,end (default: '0,-1'). "
            "Example: '0,-1 0,1 3,5 3,-1'"
        ),
    )

    args = parser.parse_args()
    experiment_folder = args.experiment_folder

    experiment_name = os.path.basename(experiment_folder)
    parent_folder = os.path.dirname(experiment_folder)
    analysis_folder = os.path.join(parent_folder, f"{experiment_name}_analysis")

    # generate the list of recordings to be processed
    recording_list = get_recording_list([analysis_folder])

    # generate the list of trans_resize.avi videos to pass to deeplabcut
    body_videos = [
        os.path.join(recording, "trans_resize.mp4") for recording in recording_list
    ]

    # run deeplabcut
    run_deeplabcut(dlc_config_path, body_videos)

    # now that done with DLC tracking, start process the recordings
    print(f"In total {len(recording_list)} videos to be processed: ")
    print(f"{[os.path.basename(recording) for recording in recording_list]}")

    # ignore warnings encountered during the process
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in arccos")
    warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_recording_wrapper, recording)
            for recording in recording_list
        ]
        # wait for completion
        concurrent.futures.wait(futures)

    # generate summary csv from the processed videos
    time_bins = parse_time_bins(args.time_bins)
    generate_summary_csv(analysis_folder, time_bins)


if __name__ == "__main__":
    main()
