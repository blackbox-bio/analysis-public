from utils import *
# from paw_luminance_rework import *

dlc_postfix = "DLC_resnet50_arcteryx500Nov4shuffle1_350000"

# Function to process a video with specified arguments
def process_recording_wrapper(recording):
    return process_recording(recording)


# Extract features from a video
#
# THIS IS AN API ENTRYPOINT! If the signature is modified, ensure api.py matches!
# The body of this function can change without affecting the API.
def extract_features(name, ftir_path, tracking_path, dest_path):
    # create a dictionary to store the extracted features
    features = {}

    # read DLC tracking
    df = pd.read_hdf(tracking_path)
    model_id = df.columns[0][0]
    label = df[model_id]

    # calculate distance traveled
    features["distance_delta"] = cal_distance_(label).reshape(-1)

    # ----calculate paw luminance, average paw luminance ratio, and paw luminance log-ratio----
    # read ftir video
    ftir_video = cv2.VideoCapture(ftir_path)
    # calculate paw luminance
    (
        hind_left,
        hind_right,
        front_left,
        front_right,
        background_luminance,
        frame_count,
    ) = cal_paw_luminance(label, ftir_video, size=22)

    fps = int(ftir_video.get(cv2.CAP_PROP_FPS))
    # recording_time = frame_count / fps

    # features["recording_time"] = np.array(recording_time)
    features["fps"] = np.array(fps)
    features["frame_count"] = np.array(frame_count)
    features["animal_detection"] = detect_animal_in_recording(label, fps)

    features["hind_left_luminance"] = hind_left
    features["hind_right_luminance"] = hind_right
    features["front_left_luminance"] = front_left
    features["front_right_luminance"] = front_right
    features["background_luminance"] = background_luminance

    # hind_left_scaled, hind_right_scaled = scale_ftir(hind_left, hind_right)
    # features["hind_left_luminance_scaled"] = hind_left_scaled
    # features["hind_right_luminance_scaled"] = hind_right_scaled
    #
    # # calculate luminance logratio
    # features["average_luminance_ratio"] = np.nansum(
    #     features["hind_left_luminance"]
    # ) / np.nansum(features["hind_right_luminance"]).reshape(-1, 1)
    # features["luminance_logratio"] = np.log(
    #     (features["hind_left_luminance_scaled"] + 1e-4)
    #     / (features["hind_right_luminance_scaled"] + 1e-4)
    # )

    # calculate when the animal is standing on two hind paws
    # features["both_front_paws_lifted"] = both_front_paws_lifted(front_left, front_right)

    # body parts distance

    # lateral body parts distance
    # lower body parts distance
    features["hip_width"] = body_parts_distance(label, "lhip", "rhip")
    features["ankle_distance"] = body_parts_distance(label, "lankle", "rankle")
    features["hind_paws_distance"] = body_parts_distance(label, "lhpaw", "rhpaw")
    # upper body parts distance
    features["shoulder_width"] = body_parts_distance(label, "lshoulder", "rshoulder")
    features["front_paws_distance"] = body_parts_distance(label, "lfpaw", "rfpaw")
    # face parts distance
    features["cheek_distance"] = body_parts_distance(label, "lcheek", "rcheek")

    # midline body parts distance
    features["tailbase_tailtip_distance"] = body_parts_distance(
        label, "tailbase", "tailtip"
    )
    features["hip_tailbase_distance"] = body_parts_distance(label, "hip", "tailbase")
    features["hip_sternumtail_distance"] = body_parts_distance(
        label, "hip", "sternumtail"
    )
    features["sternumtail_sternumhead_distance"] = body_parts_distance(
        label, "sternumtail", "sternumhead"
    )
    features["sternumhead_neck_distance"] = body_parts_distance(
        label, "sternumhead", "neck"
    )
    features["neck_snout_distance"] = body_parts_distance(label, "neck", "snout")

    # body parts vectors
    sternumtail_sternumhead_vector = get_vector(label, "sternumtail", "sternumhead")
    neck_snout_vector = get_vector(label, "neck", "snout")
    tailbase_hip_vector = get_vector(label, "tailbase", "hip")
    tailtip_tailbase_vector = get_vector(label, "tailtip", "tailbase")
    tailbase_hlpaw_vec = get_vector(label, "tailbase", "lhpaw")
    tailbase_hrpaw_vec = get_vector(label, "tailbase", "rhpaw")

    # body parts angles
    features["chest_head_angle"] = get_angle(
        neck_snout_vector, sternumtail_sternumhead_vector
    )
    features["hip_chest_angle"] = get_angle(
        sternumtail_sternumhead_vector, tailbase_hip_vector
    )
    # note the negative sign for the tail_hip_angle
    features["tail_hip_angle"] = -get_angle(
        tailbase_hip_vector, tailtip_tailbase_vector
    )
    features["hip_tailbase_hlpaw_angle"] = get_angle(
        tailbase_hip_vector, tailbase_hlpaw_vec
    )
    features["hip_tailbase_hrpaw_angle"] = get_angle(
        tailbase_hrpaw_vec, tailbase_hip_vector
    )

    # paw luminance rework!!
    ftir_video.release()
    ftir_video = cv2.VideoCapture(ftir_path)
    (paw_luminescence, paw_print_size, paw_luminance, _, _) = cal_paw_luminance_rework(
        label, ftir_video, size=22
    )

    paws = ["lhpaw", "rhpaw", "lfpaw", "rfpaw"]
    for paw in paws:
        features[f"{paw}_luminescence"] = paw_luminescence[paw]
        features[f"{paw}_print_size"] = paw_print_size[paw]
        features[f"{paw}_luminance_rework"] = paw_luminance[paw]

    # -------------------------------------------------------------

    # save extracted features
    with h5py.File(dest_path, "w") as hdf:
        video_data = hdf.create_group(name)
        for key in features.keys():
            video_data.create_dataset(key, data=features[key])


def process_recording(recording):

    print(f"Processing {os.path.basename(recording)}...")

    recording_name = os.path.basename(recording)
    ftir_path = os.path.join(recording, "ftir_resize.avi")

    dlc_postfix = "DLC_resnet50_arcteryx500Nov4shuffle1_350000"
    # dlc_path = os.path.join(recording, "trans_resize" + dlc_postfix + ".h5")

    dlc_path = os.path.join(recording, "trans_resize" + dlc_postfix + "_filtered.h5")
    dest_path = os.path.join(recording, "features.h5")

    extract_features(recording_name, ftir_path, dlc_path, dest_path)
