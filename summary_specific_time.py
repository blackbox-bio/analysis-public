from typing import Dict, Any

from utils import *

# Generate a CSV containing a summary of all features extracted from all recordings selected
#
# THIS IS AN API ENTRYPOINT! If the signature is modified, ensure api.py matches!
# The body of this function can change without affecting the API.


def time_to_frame(time, fps):
    # time in minutes
    return int(time * 60) * fps


def generate_summary_csv_specific(analysis_folder, start_time, end_time):
    """
    Generate summary csv from the processed recordings
    """
    recording_list = get_recording_list([analysis_folder])
    summary_csv = os.path.join(
        analysis_folder, f"summary_{start_time}-{end_time}min.csv"
    )

    features = defaultdict(dict)

    # read features from h5 files
    for file in [
        os.path.join(recording, "features.h5") for recording in recording_list
    ]:
        with h5py.File(file, "r") as hdf:
            for key in hdf.keys():
                for subkey in hdf[key].keys():
                    features[key][subkey] = np.array(hdf[key][subkey])

    # save summary features
    summary_features: dict[Any, dict[Any, Any]] = {}
    for video in features.keys():

        # get the start and end frame
        fps = features[video]["fps"]
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)
        if start_frame > features[video]["frame_count"]:
            print(f"Start time {start_time} is greater than the recording time.")
            continue
        if end_frame > features[video]["frame_count"]:
            print(f"End time {end_time} is greater than the recording time.")
            end_frame = features[video]["frame_count"]
            end_time = end_frame / fps / 60

        summary_features[video] = {}
        # 1. recording time
        summary_features[video]["recording_time (min)"] = end_time - start_time
        # 2. distance traveled
        summary_features[video]["distance_traveled (pixel)"] = np.nansum(
            features[video]["distance_delta"][start_frame : end_frame - 1]
        )

        # summary_features[video]["average_background_luminance"] = np.nanmean(
        #     features[video]["background_luminance"]
        # )
        # 3. standing on two hind paws
        summary_features[video][
            "standing_on_two_hind_paws (ratio of time)"
        ] = np.nanmean(features[video]["standing_on_two_paws"][start_frame:end_frame])
        # 4-7. paw luminance
        summary_features[video]["average_hind_left_luminance"] = np.nanmean(
            features[video]["hind_left_luminance"][start_frame:end_frame]
        )
        summary_features[video]["average_hind_right_luminance"] = np.nanmean(
            features[video]["hind_right_luminance"][start_frame:end_frame]
        )
        summary_features[video]["average_front_left_luminance"] = np.nanmean(
            features[video]["front_left_luminance"][start_frame:end_frame]
        )
        summary_features[video]["average_front_right_luminance"] = np.nanmean(
            features[video]["front_right_luminance"][start_frame:end_frame]
        )

        # 8-12. paw luminance ratios
        summary_features[video]["average_hind_paw_luminance_ratio (l/r)"] = (
            summary_features[video]["average_hind_left_luminance"]
            / summary_features[video]["average_hind_right_luminance"]
        )
        summary_features[video]["average_hind_paw_luminance_ratio (r/l)"] = (
            summary_features[video]["average_hind_right_luminance"]
            / summary_features[video]["average_hind_left_luminance"]
        )
        summary_features[video]["average_front_to_hind_paw_luminance_ratio"] = (
            summary_features[video]["average_front_left_luminance"]
            + summary_features[video]["average_front_right_luminance"]
        ) / (
            summary_features[video]["average_hind_left_luminance"]
            + summary_features[video]["average_hind_right_luminance"]
        )
        summary_features[video][
            "average_standing_hind_paw_luminance_ratio (l/r)"
        ] = np.nanmean(
            features[video]["hind_left_luminance"][start_frame:end_frame][
                features[video]["standing_on_two_paws"][start_frame:end_frame]
            ]
        ) / np.nanmean(
            features[video]["hind_right_luminance"][start_frame:end_frame][
                features[video]["standing_on_two_paws"][start_frame:end_frame]
            ]
        )
        summary_features[video][
            "average_standing_hind_paw_luminance_ratio (r/l)"
        ] = np.nanmean(
            features[video]["hind_right_luminance"][start_frame:end_frame][
                features[video]["standing_on_two_paws"][start_frame:end_frame]
            ]
        ) / np.nanmean(
            features[video]["hind_left_luminance"][start_frame:end_frame][
                features[video]["standing_on_two_paws"][start_frame:end_frame]
            ]
        )

        # 13-16. paw usage
        summary_features[video]["hind_left_usage (ratio of time)"] = np.nanmean(
            features[video]["hind_left_luminance"][start_frame:end_frame]
            > np.percentile(
                features[video]["background_luminance"][start_frame:end_frame], 95
            )
        )
        summary_features[video]["hind_right_usage (ratio of time)"] = np.nanmean(
            features[video]["hind_right_luminance"][start_frame:end_frame]
            > np.percentile(
                features[video]["background_luminance"][start_frame:end_frame], 95
            )
        )
        summary_features[video]["front_left_usage (ratio of time)"] = np.nanmean(
            features[video]["front_left_luminance"][start_frame:end_frame]
            > np.percentile(
                features[video]["background_luminance"][start_frame:end_frame], 95
            )
        )
        summary_features[video]["front_right_usage (ratio of time)"] = np.nanmean(
            features[video]["front_right_luminance"][start_frame:end_frame]
            > np.percentile(
                features[video]["background_luminance"][start_frame:end_frame], 95
            )
        )

    df = pd.DataFrame.from_dict(summary_features, orient="index")

    # Save DataFrame to CSV with specified precision
    df.to_csv(summary_csv, float_format="%.2f")
    return


def main():

    args = sys.argv[1:]
    assert len(args) == 2, "expecting two arguments: the start and end time in minutes"

    start_time, end_time = float(args[0]), float(args[1])
    assert start_time < end_time, "start time should be less than end time"

    # Ask the user to select subfolder to process
    root = tk.Tk()
    root.withdraw()

    # input the path to the experiment folder
    experiment_folder = select_folder()

    experiment_name = os.path.basename(experiment_folder)
    parent_folder = os.path.dirname(experiment_folder)
    analysis_folder = os.path.join(parent_folder, f"{experiment_name}_analysis")

    # generate the list of recordings to be processed
    # recording_list = get_recording_list([analysis_folder])

    # generate summary csv from the processed videos
    generate_summary_csv_specific(analysis_folder, start_time, end_time)


if __name__ == "__main__":
    import tkinter as tk
    import sys

    main()
