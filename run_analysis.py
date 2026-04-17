import palmreader_analysis
from process import *
from summary import *
from dlc_runner import *
import concurrent.futures
import sys
import warnings
import os
import argparse
from report.openfield_occupancy import plot_open_field_occupancy_map

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
        "--dlc_config_path",
        type=str,
        required=True,
        help="Path to the deeplabcut config.yaml file",
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
        )
    )
    parser.add_argument(
        "--openfield_test",
        action= "store_true",
        help=(
            "flag for running openfield test on each recording"
        )
    ),
    parser.add_argument(
        "--generate_skeleton",
        action="store_true",
        help=(
            "flag for generating skeleton video on each recording"
        )
    ),
    parser.add_argument(
        "--simple_skeleton",
        action="store_true",
        help=(
            "flag for generating simple skeleton video on each recording instead of the full skeleton"
        )
    ),

    args = parser.parse_args()
    experiment_folder = args.experiment_folder

    # generate the list of recordings to be processed
    recording_list = get_recording_list([experiment_folder])

    # generate the list of trans.avi videos to pass to deeplabcut
    body_videos = [
        os.path.join(recording, "trans.mp4") for recording in recording_list
    ]

    body_videos = list(dict.fromkeys(body_videos))

    # run deeplabcut
    run_deeplabcut(
        args.dlc_config_path,
        body_videos,
        args.generate_skeleton,
        args.simple_skeleton
    )

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

    if args.openfield_test:
        for recording in recording_list:

            features_h5 = os.path.join(recording, "features.h5")
            for file in os.listdir(recording):
                if file.endswith("_filtered.h5"):
                    dlc_path = os.path.join(recording, file)
                    break
            dest_path = os.path.join(recording, "openfield_occupancy_map.png")
            plot_open_field_occupancy_map(features_h5, dest_path)

    # generate summary csv from the processed videos
    time_bins = parse_time_bins(args.time_bins)
    generate_summary_csv(experiment_folder, time_bins)


if __name__ == "__main__":
    main()
