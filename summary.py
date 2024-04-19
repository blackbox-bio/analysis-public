from typing import Dict, Any, List, Tuple

from utils import *

def generate_summary_generic(features_files: List[str], time_bin=(0, -1)):
    features = defaultdict(dict)

    # read features from h5 files
    for file in features_files:
        with h5py.File(file, "r") as hdf:
            for key in hdf.keys():
                for subkey in hdf[key].keys():
                    features[key][subkey] = np.array(hdf[key][subkey])

    # New -> take a time bin as a tuple of start and end time in minutes
    # binning the features
    for video in features.keys():
        frame_count = features[video]["frame_count"]
        fps = features[video]["fps"]

        start_frame = int(time_bin[0] * 60 * fps)
        if time_bin[1] == -1:
            end_frame = frame_count
        else:
            end_frame = int(time_bin[1] * 60 * fps)

        # check if the time bin is valid
        if start_frame >= frame_count:
            raise ValueError(
                "Invalid time bin: start time is greater than the total recording time"
            )
        if start_frame < 0:
            raise ValueError("Invalid time bin: start time is negative")
        if end_frame > frame_count:
            # raise ValueError(
            #     "Invalid time bin: end time is greater than the total recording time"
            # )
            # if end time is greater than the total recording time, set it to the end of the recording
            end_frame = frame_count

        if end_frame < 0:
            raise ValueError("Invalid time bin: end time is negative")
        if end_frame <= start_frame:
            raise ValueError("Invalid time bin: end time is less than start time")

        # bin the features
        for key in features[video].keys():

            # change the frame count to the time bin
            if key == "frame_count":
                # features[video][key] = end_frame - start_frame
                continue

            # skip fps
            if key == "fps":
                continue

            features[video][key] = features[video][key][start_frame:end_frame]

        # add start and end time to the features
        features[video]["start_time"] = start_frame / fps / 60
        features[video]["end_time"] = end_frame / fps / 60

    # save summary features
    summary_features: dict[Any, dict[Any, Any]] = {}
    for video in features.keys():

        summary_features[video] = {}
        # 1. recording time
        summary_features[video]["total recording_time (min)"] = (
            features[video]["frame_count"] / features[video]["fps"] / 60
        )
        summary_features[video][
            "summary start_time-end_time (min)"
        ] = f'{features[video]["start_time"]:.2f}-{features[video]["end_time"]:.2f}'

        summary_features[video]["summary time duration (min)"] = (
            features[video]["end_time"] - features[video]["start_time"]
        )

        # 2. distance traveled
        summary_features[video]["distance_traveled (pixel)"] = np.nansum(
            features[video]["distance_delta"]
        )

        # summary_features[video]["average_background_luminance"] = np.nanmean(
        #     features[video]["background_luminance"]
        # )
        # 3. both_front_paws_lifted
        summary_features[video]["both_front_paws_lifted (ratio of time)"] = np.nanmean(
            features[video]["both_front_paws_lifted"]
        )
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
        summary_features[video]["average_all_paws_sum_luminance"] = (
            np.nanmean(features[video]["hind_left_luminance"])
            + np.nanmean(features[video]["hind_right_luminance"])
            + np.nanmean(features[video]["front_left_luminance"])
            + np.nanmean(features[video]["front_right_luminance"])
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
                features[video]["both_front_paws_lifted"]
            ]
        ) / np.nanmean(
            features[video]["hind_right_luminance"][
                features[video]["both_front_paws_lifted"]
            ]
        )
        summary_features[video][
            "average_standing_hind_paw_luminance_ratio (r/l)"
        ] = np.nanmean(
            features[video]["hind_right_luminance"][
                features[video]["both_front_paws_lifted"]
            ]
        ) / np.nanmean(
            features[video]["hind_left_luminance"][
                features[video]["both_front_paws_lifted"]
            ]
        )

        # # 13-16. paw usage
        # summary_features[video]["hind_left_usage (ratio of time)"] = np.nanmean(
        #     features[video]["hind_left_luminance"] > 1e-4
        # )
        # summary_features[video]["hind_right_usage (ratio of time)"] = np.nanmean(
        #     features[video]["hind_right_luminance"] > 1e-4
        # )
        # summary_features[video]["front_left_usage (ratio of time)"] = np.nanmean(
        #     features[video]["front_left_luminance"] > 1e-4
        # )
        # summary_features[video]["front_right_usage (ratio of time)"] = np.nanmean(
        #     features[video]["front_right_luminance"] > 1e-4
        # )

        # 17-20 time spent paw lifted (not touching the ground)
        summary_features[video]["hind_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["hind_left_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["hind_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["hind_right_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["front_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["front_left_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["front_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["front_right_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        # body parts distance
        # 21-26. lateral body parts distance
        summary_features[video]["hip_width (pixel)"] = np.nanmean(
            features[video]["hip_width"]
        )
        summary_features[video]["ankle_distance (pixel)"] = np.nanmean(
            features[video]["ankle_distance"]
        )
        summary_features[video]["hind_paws_distance (pixel)"] = np.nanmean(
            features[video]["hind_paws_distance"]
        )
        summary_features[video]["shoulder_width (pixel)"] = np.nanmean(
            features[video]["shoulder_width"]
        )
        summary_features[video]["front_paws_distance (pixel)"] = np.nanmean(
            features[video]["front_paws_distance"]
        )
        summary_features[video]["cheek_distance (pixel)"] = np.nanmean(
            features[video]["cheek_distance"]
        )
        # 27-32. midline body parts distance
        summary_features[video]["tailbase_tailtip_distance (pixel)"] = np.nanmean(
            features[video]["tailbase_tailtip_distance"]
        )
        summary_features[video]["hip_tailbase_distance (pixel)"] = np.nanmean(
            features[video]["hip_tailbase_distance"]
        )
        summary_features[video]["hip_sternumtail_distance (pixel)"] = np.nanmean(
            features[video]["hip_sternumtail_distance"]
        )
        summary_features[video][
            "sternumtail_sternumhead_distance (pixel)"
        ] = np.nanmean(features[video]["sternumtail_sternumhead_distance"])
        summary_features[video]["sternumhead_neck_distance (pixel)"] = np.nanmean(
            features[video]["sternumhead_neck_distance"]
        )
        summary_features[video]["neck_snout_distance (pixel)"] = np.nanmean(
            features[video]["neck_snout_distance"]
        )

        # 33-35. midline body parts angles
        summary_features[video]["chest_head_angle (degree)"] = np.nanmean(
            features[video]["chest_head_angle"]
        )
        summary_features[video]["hip_chest_angle (degree)"] = np.nanmean(
            features[video]["hip_chest_angle"]
        )
        summary_features[video]["tail_hip_angle (degree)"] = np.nanmean(
            features[video]["tail_hip_angle"]
        )

        # 36-37. hind paw angles
        summary_features[video]["hip_tailbase_hlpaw_angle (degree)"] = np.nanmean(
            features[video]["hip_tailbase_hlpaw_angle"]
        )
        summary_features[video]["hip_tailbase_hrpaw_angle (degree)"] = np.nanmean(
            features[video]["hip_tailbase_hrpaw_angle"]
        )

    df = pd.DataFrame.from_dict(summary_features, orient="index")

    # # Save DataFrame to CSV with specified precision
    # df.to_csv(summary_dest, float_format="%.2f")
    return df

def _df_concat_step(prev, next):
    if prev is None:
        return next
    
    return pd.concat([prev, next])

def generate_summaries_generic(features_files: List[str], time_bins: List[Tuple[float, float]]):
    df = None

    for time_bin in time_bins:
        df = _df_concat_step(df, generate_summary_generic(features_files, time_bin))
    
    return df

def generate_summary_csv(analysis_folder, time_bins):
    """
    Generate summary csv from the processed recordings
    """
    recording_list = get_recording_list([analysis_folder])
    summary_dest = os.path.join(analysis_folder, "summary.csv")

    features_files = [
        os.path.join(recording, "features.h5") for recording in recording_list
    ]

    df = generate_summaries_generic(features_files, time_bins)

    df.to_csv(summary_dest, float_format="%.2f")
