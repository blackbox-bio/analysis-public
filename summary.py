from typing import Dict, Any

from utils import *


def generate_summary_csv(features_folder, summary_csv):
    """
    Generate summary csv from the processed videos
    """
    features = defaultdict(dict)

    # read features from h5 files
    for file in tqdm(os.listdir(features_folder)):
        if ".h5" in file:
            with h5py.File(os.path.join(features_folder, file), "r") as hdf:
                for key in hdf.keys():
                    for subkey in hdf[key].keys():
                        features[key][subkey] = np.array(hdf[key][subkey])

    # save summary features
    summary_features: dict[Any, dict[Any, Any]] = {}
    for video in features.keys():
        summary_features[video] = {}
        summary_features[video]["recording_time (min)"] = (
            features[video]["recording_time"] / 60
        )
        summary_features[video]["distance_traveled (pixel)"] = np.nansum(
            features[video]["distance_traveled"]
        )

        # paw luminance
        # summary_features[video]["average_background_luminance"] = np.nanmean(
        #     features[video]["background_luminance"]
        # )
        summary_features[video][
            "standing_on_two_hind_paws (ratio of time)"
        ] = np.nanmean(features[video]["standing_on_two_paws"])
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

        # paw luminance ratios
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

        # paw usage
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
    df.to_csv(summary_csv, float_format="%.2f")
    return


# # save summary features
#     summary_features = {
#         "video_name": video,
#         "recording_time (min)": recording_time / 60,
#         "distance_traveled (cm)": np.nansum(features["distance_traveled"]) / 1024 * 15,
#         "average_hind_paw_luminance_ratio (l/r)": np.nanmean(
#             features["average_luminance_ratio"]
#         ),
#         "average_hind_paw_luminance_ratio (r/l)": np.nanmean(
#             1 / features["average_luminance_ratio"]
#         ),
#     }
#     df = pd.DataFrame.from_dict([summary_features])
#     summary_csv = os.path.join(output_folder, "summary.csv")
#
#     # Save DataFrame to CSV with specified precision
#     df.to_csv(summary_csv, index=False, float_format="%.2f")
#     return
