from typing import Dict, Any, List, Tuple

from utils import *
from cols_name_dicts import *
from palmreader_analysis.summary import SummaryContext


def generate_summary_generic(features_files: List[str], time_bin=(0, -1)):
    features = defaultdict(dict)

    # temp: summary contexts
    summary_contexts: Dict[str, SummaryContext] = {}

    # read features from h5 files
    for file in features_files:
        with h5py.File(file, "r") as hdf:
            for key in hdf.keys():
                # note: this does not do what is expected when there are multiple recordings in the h5 file. Palmreader does not do that.
                summary_contexts[key] = SummaryContext(file, time_bin)
                for subkey in hdf[key].keys():
                    features[key][subkey] = np.array(hdf[key][subkey])

    # trim the recording length by automatic animal detection
    for video in features.keys():
        if "animal_detection" in features[video].keys():
            animal_detection = features[video]["animal_detection"]
            frame_count = features[video]["frame_count"]
            fps = features[video]["fps"]
            start_frame = 0
            end_frame = frame_count  # default to the end of the recording

            # find the start frame
            for i in range(frame_count):
                if animal_detection[i] == 1:
                    start_frame = i
                    break

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
            features[video]["frame_count"] = end_frame - start_frame

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
            "PV: bin start-end (min)"
        ] = f'{features[video]["start_time"]:.2f} - {features[video]["end_time"]:.2f}'

        summary_features[video]["bin duration (min)"] = (
            features[video]["end_time"] - features[video]["start_time"]
        )

        # 2. distance traveled
        summary_features[video]["distance_traveled (pixel)"] = np.nansum(
            features[video]["distance_delta"]
        )

        # summary_features[video]["average_background_luminance"] = np.nanmean(
        #     features[video]["background_luminance"]
        # )

        # paw luminance rework!!
        paws = ["lhpaw", "rhpaw", "lfpaw", "rfpaw"]
        paws_dict = {
            "lhpaw": "hind_left",
            "rhpaw": "hind_right",
            "lfpaw": "front_left",
            "rfpaw": "front_right",
        }
        # lum_quant = ["luminescence", "print_size", "luminance_rework"]
        # lum_quant = ["luminescence", "print_size",]

        # paw luminescence/print/luminance for internal use
        lh_luminescence = np.nanmean(features[video]["lhpaw_luminescence"])
        rh_luminescence = np.nanmean(features[video]["rhpaw_luminescence"])
        lf_luminescence = np.nanmean(features[video]["lfpaw_luminescence"])
        rf_luminescence = np.nanmean(features[video]["rfpaw_luminescence"])
        lf_print = np.nanmean(features[video]["lfpaw_print_size"])
        rf_print = np.nanmean(features[video]["rfpaw_print_size"])
        lh_print = np.nanmean(features[video]["lhpaw_print_size"])
        rh_print = np.nanmean(features[video]["rhpaw_print_size"])
        lh_luminance = np.nanmean(features[video]["lhpaw_luminance_rework"])
        rh_luminance = np.nanmean(features[video]["rhpaw_luminance_rework"])
        lf_luminance = np.nanmean(features[video]["lfpaw_luminance_rework"])
        rf_luminance = np.nanmean(features[video]["rfpaw_luminance_rework"])

        quant = "print_size"
        summary_features[video]["average_overall_print_size (pixel area)"] = (
            lf_print + rf_print + lh_print + rh_print
        )
        for paw in paws:
            summary_features[video][
                f"average_{paws_dict[paw]}_{quant} (pixel area)"
            ] = np.nanmean(features[video][f"{paw}_{quant}"])
            summary_features[video][f"relative_{paws_dict[paw]}_{quant} (ratio)"] = (
                np.nanmean(features[video][f"{paw}_{quant}"])
                / summary_features[video]["average_overall_print_size (pixel area)"]
            )

        quant = "luminescence"
        summary_features[video]["average_overall_luminescence (pixel intensity)"] = (
            lf_luminescence + rf_luminescence + lh_luminescence + rh_luminescence
        )
        for paw in paws:
            summary_features[video][
                f"average_{paws_dict[paw]}_{quant} (pixel intensity)"
            ] = np.nanmean(features[video][f"{paw}_{quant}"])
            summary_features[video][f"relative_{paws_dict[paw]}_{quant} (ratio)"] = (
                np.nanmean(features[video][f"{paw}_{quant}"])
                / summary_features[video][
                    "average_overall_luminescence (pixel intensity)"
                ]
            )

        quant = "luminance_rework"
        summary_features[video]["average_overall_luminance (pixel intensity/area)"] = (
            lf_luminance + rf_luminance + lh_luminance + rh_luminance
        )
        for paw in paws:
            summary_features[video][
                f"average_{paws_dict[paw]}_luminance (pixel intensity/area)"
            ] = np.nanmean(features[video][f"{paw}_{quant}"])
            summary_features[video][f"relative_{paws_dict[paw]}_luminance (ratio)"] = (
                np.nanmean(features[video][f"{paw}_{quant}"])
                / summary_features[video][
                    "average_overall_luminance (pixel intensity/area)"
                ]
            )

        # 8-12. paw luminescence/print/luminance ratios
        summary_features[video]["average_hind_paw_luminescence_ratio (l/r)"] = (
            lh_luminescence / rh_luminescence
        )
        summary_features[video]["average_hind_paw_luminescence_ratio (r/l)"] = (
            rh_luminescence / lh_luminescence
        )
        summary_features[video]["average_front_to_hind_paw_luminescence_ratio"] = (
            lf_luminescence + rf_luminescence
        ) / (lh_luminescence + rh_luminescence)

        # calculate a boolean array for standing (both front paws lifted)
        standing = both_front_paws_lifted(
            features[video]["lfpaw_luminance_rework"],
            features[video]["rfpaw_luminance_rework"],
        )

        summary_features[video][
            "average_standing_hind_paw_luminescence_ratio (l/r)"
        ] = np.nanmean(
            features[video]["lhpaw_luminance_rework"][standing]
        ) / np.nanmean(
            features[video]["rhpaw_luminance_rework"][standing]
        )
        summary_features[video]["average_standing_hind_paw_luminance_ratio (r/l)"] = (
            np.nanmean(features[video]["rhpaw_luminance_rework"][standing])
            / np.nanmean(features[video]["lhpaw_luminance_rework"][standing])
        )

        summary_features[video]["average_hind_paw_print_size_ratio (l/r)"] = (
            lh_print / rh_print
        )
        summary_features[video]["average_hind_paw_print_size_ratio (r/l)"] = (
            rh_print / lh_print
        )
        summary_features[video]["average_front_to_hind_paw_print_size_ratio"] = (
            lf_print + rf_print
        ) / (lh_print + rh_print)

        summary_features[video]["average_standing_hind_paw_print_size_ratio (l/r)"] = (
            np.nanmean(features[video]["lhpaw_print_size"][standing])
            / np.nanmean(features[video]["rhpaw_print_size"][standing])
        )
        summary_features[video]["average_standing_hind_paw_print_size_ratio (r/l)"] = (
            np.nanmean(features[video]["rhpaw_print_size"][standing])
            / np.nanmean(features[video]["lhpaw_print_size"][standing])
        )

        summary_features[video]["average_hind_paw_luminance_ratio (l/r)"] = (
            lh_luminance / rh_luminance
        )
        summary_features[video]["average_hind_paw_luminance_ratio (r/l)"] = (
            rh_luminance / lh_luminance
        )
        summary_features[video]["average_front_to_hind_paw_luminance_ratio"] = (
            lf_luminance + rf_luminance
        ) / (lh_luminance + rh_luminance)

        summary_features[video]["average_standing_hind_paw_luminance_ratio (l/r)"] = (
            np.nanmean(features[video]["lhpaw_luminance_rework"][standing])
            / np.nanmean(features[video]["rhpaw_luminance_rework"][standing])
        )
        summary_features[video]["average_standing_hind_paw_luminance_ratio (r/l)"] = (
            np.nanmean(features[video]["rhpaw_luminance_rework"][standing])
            / np.nanmean(features[video]["lhpaw_luminance_rework"][standing])
        )

        # 17-20 time spent paw lifted (not touching the ground)
        summary_features[video]["hind_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["lhpaw_luminance_rework"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["hind_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["rhpaw_luminance_rework"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["front_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["lfpaw_luminance_rework"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["front_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["rfpaw_luminance_rework"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["both_front_paws_lifted (seconds)"] = (
            np.sum(standing) / features[video]["fps"]
        )

        # ------------legacy --------

        # 4-7. paw luminance
        summary_features[video]["legacy: average_hind_left_luminance"] = np.nanmean(
            features[video]["hind_left_luminance"]
        )
        summary_features[video]["legacy: average_hind_right_luminance"] = np.nanmean(
            features[video]["hind_right_luminance"]
        )
        summary_features[video]["legacy: average_hind_right_luminance"] = np.nanmean(
            features[video]["hind_right_luminance"]
        )
        summary_features[video]["legacy: average_front_left_luminance"] = np.nanmean(
            features[video]["front_left_luminance"]
        )
        summary_features[video]["legacy: average_front_right_luminance"] = np.nanmean(
            features[video]["front_right_luminance"]
        )
        summary_features[video]["legacy: average_all_paws_sum_luminance"] = (
            np.nanmean(features[video]["hind_left_luminance"])
            + np.nanmean(features[video]["hind_right_luminance"])
            + np.nanmean(features[video]["front_left_luminance"])
            + np.nanmean(features[video]["front_right_luminance"])
        )
        # paw luminance normalized by sum of paw luminance
        summary_features[video]["legacy: relative_hind_left_luminance"] = (
            summary_features[video]["legacy: average_hind_left_luminance"]
            / summary_features[video]["legacy: average_all_paws_sum_luminance"]
        )
        summary_features[video]["legacy: relative_hind_right_luminance"] = (
            summary_features[video]["legacy: average_hind_right_luminance"]
            / summary_features[video]["legacy: average_all_paws_sum_luminance"]
        )
        summary_features[video]["legacy: relative_front_left_luminance"] = (
            summary_features[video]["legacy: average_front_left_luminance"]
            / summary_features[video]["legacy: average_all_paws_sum_luminance"]
        )
        summary_features[video]["legacy: relative_front_right_luminance"] = (
            summary_features[video]["legacy: average_front_right_luminance"]
            / summary_features[video]["legacy: average_all_paws_sum_luminance"]
        )
        # 8-12. paw luminance ratios
        summary_features[video]["legacy: average_hind_paw_luminance_ratio (l/r)"] = (
            summary_features[video]["legacy: average_hind_left_luminance"]
            / summary_features[video]["legacy: average_hind_right_luminance"]
        )
        summary_features[video]["legacy: average_hind_paw_luminance_ratio (r/l)"] = (
            summary_features[video]["legacy: average_hind_right_luminance"]
            / summary_features[video]["legacy: average_hind_left_luminance"]
        )
        summary_features[video]["legacy: average_front_to_hind_paw_luminance_ratio"] = (
            summary_features[video]["legacy: average_front_left_luminance"]
            + summary_features[video]["legacy: average_front_right_luminance"]
        ) / (
            summary_features[video]["legacy: average_hind_left_luminance"]
            + summary_features[video]["legacy: average_hind_right_luminance"]
        )

        standing_ratio_legacy = both_front_paws_lifted(
            features[video]["front_left_luminance"],
            features[video]["front_right_luminance"],
        )

        summary_features[video][
            "legacy: average_standing_hind_paw_luminance_ratio (l/r)"
        ] = np.nanmean(
            features[video]["hind_left_luminance"][standing_ratio_legacy]
        ) / np.nanmean(
            features[video]["hind_right_luminance"][standing_ratio_legacy]
        )
        summary_features[video][
            "legacy: average_standing_hind_paw_luminance_ratio (r/l)"
        ] = np.nanmean(
            features[video]["hind_right_luminance"][standing_ratio_legacy]
        ) / np.nanmean(
            features[video]["hind_left_luminance"][standing_ratio_legacy]
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
        summary_features[video]["legacy: hind_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["hind_left_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["legacy: hind_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["hind_right_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["legacy: front_left_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["front_left_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        summary_features[video]["legacy: front_right_paw_lifted_time (seconds)"] = (
            np.sum(features[video]["front_right_luminance"] < 1e-4)
            / features[video]["fps"]
        )
        # 3. both_front_paws_lifted

        summary_features[video]["legacy: both_front_paws_lifted (ratio of time)"] = (
            np.nanmean(standing_ratio_legacy)
        )
        # ------------legacy -------- end

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
        summary_features[video]["sternumtail_sternumhead_distance (pixel)"] = (
            np.nanmean(features[video]["sternumtail_sternumhead_distance"])
        )
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

        # new toe spread and paw length for both hind paws
        summary_features[video]["hind_left_toes_spread (pixel)"] = np.nanmean(
            features[video]["hind_left_toes_spread"]
        )
        summary_features[video]["hind_right_toes_spread (pixel)"] = np.nanmean(
            features[video]["hind_right_toes_spread"]
        )
        summary_features[video]["hind_left_paw_length (pixel)"] = np.nanmean(
            features[video]["hind_left_paw_length"]
        )
        summary_features[video]["hind_right_paw_length (pixel)"] = np.nanmean(
            features[video]["hind_right_paw_length"]
        )

        # new paw angles with respect to midline for both hind paws
        summary_features[video]["hind_left_paw_angle (degree)"] = np.nanmean(
            features[video]["midline_hlpaw_angle"]
        )
        summary_features[video]["hind_right_paw_angle (degree)"] = np.nanmean(
            features[video]["midline_hrpaw_angle"]
        )

        # # new toe angles for both hind paws
        # summary_features[video]["lhpd1t_lankle_lhpaw_angle (degree)"] = np.nanmean(
        #     features[video]["lhpd1t_lankle_lhpaw_angle"]
        # )
        # summary_features[video]["lhpd5t_lankle_lhpaw_angle (degree)"] = np.nanmean(
        #     features[video]["lhpd5t_lankle_lhpaw_angle"]
        # )
        # summary_features[video]["rhpd1t_rankle_rhpaw_angle (degree)"] = np.nanmean(
        #     features[video]["rhpd1t_rankle_rhpaw_angle"]
        # )
        # summary_features[video]["rhpd5t_rankle_rhpaw_angle (degree)"] = np.nanmean(
        #     features[video]["rhpd5t_rankle_rhpaw_angle"]
        # )

        # average tracking likelihood for each paws
        summary_features[video]["average_lhpaw_tracking_likelihood"] = np.nanmean(
            features[video]["lhpaw_tracking_likelihood"]
        )
        summary_features[video]["average_rhpaw_tracking_likelihood"] = np.nanmean(
            features[video]["rhpaw_tracking_likelihood"]
        )
        summary_features[video]["average_lfpaw_tracking_likelihood"] = np.nanmean(
            features[video]["lfpaw_tracking_likelihood"]
        )
        summary_features[video]["average_rfpaw_tracking_likelihood"] = np.nanmean(
            features[video]["rfpaw_tracking_likelihood"]
        )

        # average tracking likelihood for key central line body parts
        # summary_features[video]["average_hip_tracking_likelihood"] = np.nanmean(
        #     features[video]["hip_tracking_likelihood"]
        # )
        # summary_features[video]["average_tailbase_tracking_likelihood"] = np.nanmean(
        #     features[video]["tailbase_tracking_likelihood"]
        # )
        # summary_features[video]["average_snout_tracking_likelihood"] = np.nanmean(
        #     features[video]["snout_tracking_likelihood"]
        # )

        # paws tracking quality control flag
        # 0: good, 1: bad
        # hind paws tracking likelihood need to be higher than 0.85
        # front paws tracking likelihood need to be higher than 0.6
        summary_features[video]["paws_tracking_quality_control_flag"] = 0
        if (
            summary_features[video]["average_lhpaw_tracking_likelihood"] < 0.85
            or summary_features[video]["average_rhpaw_tracking_likelihood"] < 0.85
            or summary_features[video]["average_lfpaw_tracking_likelihood"] < 0.6
            or summary_features[video]["average_rfpaw_tracking_likelihood"] < 0.6
        ):
            summary_features[video]["paws_tracking_quality_control_flag"] = 1

        # change column names for the summary to be more readable
        summary_features[video] = {
            summary_col_name_dict[k]: v for k, v in summary_features[video].items()
        }

        context = summary_contexts[video]

        for column in SummaryContext.get_all_columns():
            column.summarize(context)

        context.finish()

        context.compare_summary_columns(summary_features[video])

    df = pd.DataFrame.from_dict(summary_features, orient="index")

    # # Save DataFrame to CSV with specified precision
    # df.to_csv(summary_dest, float_format="%.2f")
    return df


def _df_concat_step(prev, next):
    if prev is None:
        return next

    return pd.concat([prev, next])


def generate_summaries_generic(
    features_files: List[str], time_bins: List[Tuple[float, float]]
):
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
