from typing import Dict, Any, List

from utils import *

def generate_summary_v1(features_folder: str, summary_dest: str):
    """
    v1 API expects all features to be in a single folder. this function collects all .h5 files in the given folder and uses them
    """
    features_files = []

    for file in os.listdir(features_folder):
        if file.endswith(".h5"):
            features_files.append(os.path.join(features_folder, file))

    generate_summary_generic(features_files, summary_dest)

def generate_summary_v2(features_files: List[str], summary_dest: str):
    """
    v2 API expects a list of .h5 files. this function uses them directly
    """
    generate_summary_generic(features_files, summary_dest)

def generate_summary_generic(features_files: List[str], summary_dest: str):
    features = defaultdict(dict)

    # read features from h5 files
    for file in features_files:
        with h5py.File(file, "r") as hdf:
            for key in hdf.keys():
                for subkey in hdf[key].keys():
                    features[key][subkey] = np.array(hdf[key][subkey])

    # save summary features
    summary_features: dict[Any, dict[Any, Any]] = {}
    for video in features.keys():
        summary_features[video] = {}
        # 1. recording time
        summary_features[video]["recording_time (min)"] = (
            features[video]["recording_time"] / 60
        )
        # 2. distance traveled
        summary_features[video]["distance_traveled (pixel)"] = np.nansum(
            features[video]["distance_delta"]
        )

        # summary_features[video]["average_background_luminance"] = np.nanmean(
        #     features[video]["background_luminance"]
        # )
        # 3. standing on two hind paws
        summary_features[video][
            "standing_on_two_hind_paws (ratio of time)"
        ] = np.nanmean(features[video]["standing_on_two_paws"])
        # 4-7. paw luminance
        summary_features[video]["average_hind_left_luminance"] = np.nanmean(
            features[video]["hind_left_luminance"]
        )
        summary_features[video]["average_hind_right_luminance"] = np.nanmean(
            features[video]["hind_right_luminance"]
        )
        summary_features[video]["average_front_left_luminance"] = np.nanmean(
            features[video]["front_left_luminance"]
        )
        summary_features[video]["average_front_right_luminance"] = np.nanmean(
            features[video]["front_right_luminance"]
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
            features[video]["hind_left_luminance"][
                features[video]["standing_on_two_paws"]
            ]
        ) / np.nanmean(
            features[video]["hind_right_luminance"][
                features[video]["standing_on_two_paws"]
            ]
        )
        summary_features[video][
            "average_standing_hind_paw_luminance_ratio (r/l)"
        ] = np.nanmean(
            features[video]["hind_right_luminance"][
                features[video]["standing_on_two_paws"]
            ]
        ) / np.nanmean(
            features[video]["hind_left_luminance"][
                features[video]["standing_on_two_paws"]
            ]
        )

        # 13-16. paw usage
        summary_features[video]["hind_left_usage (ratio of time)"] = np.nanmean(
            features[video]["hind_left_luminance"]
            > np.percentile(features[video]["background_luminance"], 95)
        )
        summary_features[video]["hind_right_usage (ratio of time)"] = np.nanmean(
            features[video]["hind_right_luminance"]
            > np.percentile(features[video]["background_luminance"], 95)
        )
        summary_features[video]["front_left_usage (ratio of time)"] = np.nanmean(
            features[video]["front_left_luminance"]
            > np.percentile(features[video]["background_luminance"], 95)
        )
        summary_features[video]["front_right_usage (ratio of time)"] = np.nanmean(
            features[video]["front_right_luminance"]
            > np.percentile(features[video]["background_luminance"], 95)
        )

    df = pd.DataFrame.from_dict(summary_features, orient="index")

    # Save DataFrame to CSV with specified precision
    df.to_csv(summary_dest, float_format="%.2f")
    return

def generate_summary_csv(analysis_folder):
    """
    Generate summary csv from the processed recordings
    """
    recording_list = get_recording_list([analysis_folder])
    summary_csv = os.path.join(analysis_folder, "summary.csv")

    features_files = [os.path.join(recording, "features.h5") for recording in recording_list]

    generate_summary_generic(features_files, summary_csv)
