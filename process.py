from utils import *
from palmreader_analysis.features import FeaturesContext


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

    (
        paw_luminescence,
        paw_print_size,
        paw_luminance,
        background_luminance,
        frame_count,
        legacy_paw_luminance,
    ) = cal_paw_luminance_rework(label, ftir_video, size=22)

    paws = ["lhpaw", "rhpaw", "lfpaw", "rfpaw"]
    for paw in paws:
        features[f"{paw}_luminescence"] = paw_luminescence[paw]
        features[f"{paw}_print_size"] = paw_print_size[paw]
        features[f"{paw}_luminance_rework"] = paw_luminance[paw]

    # calculate paw luminance
    # (
    #     hind_left,
    #     hind_right,
    #     front_left,
    #     front_right,
    #     background_luminance,
    #     frame_count,
    # ) = cal_paw_luminance(label, ftir_video, size=22)

    (
        hind_left,
        hind_right,
        front_left,
        front_right,
    ) = legacy_paw_luminance
    #
    fps = int(ftir_video.get(cv2.CAP_PROP_FPS))

    ## recording_time = frame_count / fps
    ##
    ## features["recording_time"] = np.array(recording_time)

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

    # toe spread and paw length for both hind paws
    features["hind_left_toes_spread"] = body_parts_distance(label, "lhpd1t", "lhpd5t")
    features["hind_right_toes_spread"] = body_parts_distance(label, "rhpd1t", "rhpd5t")
    features["hind_left_paw_length"] = body_parts_distance(label, "lankle", "lhpd3t")
    features["hind_right_paw_length"] = body_parts_distance(label, "rankle", "rhpd3t")

    # body parts vectors
    sternumtail_sternumhead_vector = get_vector(label, "sternumtail", "sternumhead")
    midline_vector = get_vector(label, "tailbase", "sternumtail")
    neck_snout_vector = get_vector(label, "neck", "snout")
    tailbase_hip_vector = get_vector(label, "tailbase", "hip")
    tailtip_tailbase_vector = get_vector(label, "tailtip", "tailbase")
    tailbase_hlpaw_vec = get_vector(label, "tailbase", "lhpaw")
    tailbase_hrpaw_vec = get_vector(label, "tailbase", "rhpaw")
    lankle_lhpaw_vec = get_vector(label, "lankle", "lhpaw")
    lankle_lhpd1t_vec = get_vector(label, "lankle", "lhpd1t")
    # lankle_lhpd3t_vec = get_vector(label,"lankle","lhpd3t")
    lankle_lhpd5t_vec = get_vector(label, "lankle", "lhpd5t")
    rankle_rhpaw_vec = get_vector(label, "rankle", "rhpaw")
    rankle_rhpd1t_vec = get_vector(label, "rankle", "rhpd1t")
    # rankle_rhpd3t_vec = get_vector(label,"rankle","rhpd3t")
    rankle_rhpd5t_vec = get_vector(label, "rankle", "rhpd5t")

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
    # paw angles with respect to the midline for both hind paws
    features["midline_hlpaw_angle"] = get_angle(midline_vector, lankle_lhpaw_vec)
    features["midline_hrpaw_angle"] = get_angle(rankle_rhpaw_vec, midline_vector)

    # # toe angles for both hind paws
    # features["lhpd1t_lankle_lhpaw_angle"] = get_angle(
    #     lankle_lhpd1t_vec, lankle_lhpaw_vec
    # )
    # features["lhpd5t_lankle_lhpaw_angle"] = get_angle(
    #     lankle_lhpaw_vec, lankle_lhpd5t_vec
    # )
    # features["rhpd1t_rankle_rhpaw_angle"] = get_angle(
    #     rankle_rhpaw_vec, rankle_rhpd1t_vec
    # )
    # features["rhpd5t_rankle_rhpaw_angle"] = get_angle(
    #     rankle_rhpd5t_vec, lankle_lhpaw_vec
    # )

    # tracking likelihood for each paws
    features["lhpaw_tracking_likelihood"] = label["lhpaw"]["likelihood"]
    features["rhpaw_tracking_likelihood"] = label["rhpaw"]["likelihood"]
    features["lfpaw_tracking_likelihood"] = label["lfpaw"]["likelihood"]
    features["rfpaw_tracking_likelihood"] = label["rfpaw"]["likelihood"]

    # tracking likelihood for key central line body parts
    # features["hip_tracking_likelihood"] = label["hip"]["likelihood"]
    # features["tailbase_tracking_likelihood"] = label["tailbase"]["likelihood"]
    # features["snout_tracking_likelihood"] = label["snout"]["likelihood"]

    ftir_video.release()

    # temp: OOP computation + comparison to legacy implementation; will replace
    # legacy
    ctx = FeaturesContext(name, tracking_path, ftir_path)

    for feature in FeaturesContext.get_all_features():
        feature.extract(ctx=ctx)

    ctx.compare_feature_tables(features)
    ctx.to_hdf5(f"{dest_path}_test")

    # -------------------------------------------------------------

    # save extracted features
    # print(f"Saving features to {dest_path}...")
    with h5py.File(dest_path, "w") as hdf:
        video_data = hdf.create_group(name)
        for key in features.keys():
            video_data.create_dataset(key, data=features[key])


def extract_posture_with_moving_background(
    recording_name, trans_path, ftir_path, dlc_path
):

    print(f"Processing {recording_name}...")

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    output_trans_path = os.path.join(
        os.path.dirname(trans_path), "trans_posture_with_bg.mp4"
    )
    output_ftir_path = os.path.join(
        os.path.dirname(ftir_path), "ftir_posture_with_bg.mp4"
    )

    # Load tracking data
    tracking_df = pd.read_hdf(dlc_path)
    model_id = tracking_df.columns[0][0]
    label = tracking_df[model_id]

    # Calculate orientation vector and center
    orientation_vector = cal_orientation_vector(label)  # Shape: (N, 2)
    center = label["sternumtail"][["x", "y"]].values  # Shape: (N, 2)
    center = center[: orientation_vector.shape[0]]  # Match length

    # Calculate speed along the orientation vector
    displacement = center[1:] - center[:-1]  # Displacement between consecutive frames
    speed_along_orientation = np.einsum(
        "ij,ij->i", displacement, orientation_vector[:-1]
    )  # Dot product
    speed_along_orientation = np.insert(
        speed_along_orientation, 0, 0
    )  # Insert 0 for the first frame

    # Calculate speed perpendicular to the orientation vector
    perpendicular_velocity = np.einsum(
        "ij,ij->i", displacement, np.flip(orientation_vector[:-1], axis=1) * [-1, 1]
    )
    perpendicular_velocity = np.insert(
        perpendicular_velocity, 0, 0
    )  # Insert 0 for the first frame

    # Open video files
    input_trans_cap = cv2.VideoCapture(trans_path)
    input_ftir_cap = cv2.VideoCapture(ftir_path)
    if not input_trans_cap.isOpened() or not input_ftir_cap.isOpened():
        print("Error: Video file(s) not opened")
        return

    # Get video properties
    fps = int(input_trans_cap.get(cv2.CAP_PROP_FPS))
    frame_ht = int(input_trans_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_wd = int(input_trans_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_num = int(input_trans_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_num = min(frame_num, orientation_vector.shape[0])

    # compare frame count with the length of the orientation vector, and take the minimum
    if frame_num > orientation_vector.shape[0]:
        frame_num = orientation_vector.shape[0]
        print(
            "Warning: The orientation vector is shorter than the video, which means the last few frames in the video were corrupted and will not be processed."
        )

    # Create VideoWriter objects
    output_trans = cv2.VideoWriter(output_trans_path, fourcc, fps, (frame_wd, frame_ht))
    output_ftir = cv2.VideoWriter(output_ftir_path, fourcc, fps, (frame_wd, frame_ht))

    # Create separate background patterns
    bg_pattern_horizontal = np.zeros((frame_ht, frame_wd, 3), dtype=np.uint8)
    bg_pattern_vertical = np.zeros((frame_ht, frame_wd, 3), dtype=np.uint8)

    stripe_thickness = int(frame_ht / 16)  # Thickness of each stripe
    stripe_spacing = int(frame_ht / 4)  # Spacing between stripes
    left_edge_width = int(frame_wd * 0.35)  # Width of the left bar area
    right_edge_start = int(frame_wd * 0.65)  # Start of the right bar area
    top_bar_height = int(frame_ht * 0.25)  # Height of the top bar area
    bottom_bar_start = int(frame_ht * 0.75)  # Start of the bottom bar area

    # Horizontal stripes in the left and right areas
    for y in range(0, frame_ht, stripe_spacing):
        bg_pattern_horizontal[y : y + stripe_thickness, :left_edge_width] = (
            128,
            128,
            128,
        )
        bg_pattern_horizontal[y : y + stripe_thickness, right_edge_start:] = (
            128,
            128,
            128,
        )

    # Vertical stripes in the top and bottom areas
    for x in range(0, frame_wd, stripe_spacing):
        bg_pattern_vertical[:top_bar_height, x : x + stripe_thickness] = (128, 128, 128)
        bg_pattern_vertical[bottom_bar_start:, x : x + stripe_thickness] = (
            128,
            128,
            128,
        )

    # Adjusted line properties
    line_color = (128, 128, 128)  # Gray color for the lines
    line_thickness = 4
    line_offset = 4  # Offset to ensure lines are within the main animal frame

    # Adjusted positions for the lines
    cv2.line(
        bg_pattern_horizontal,
        (left_edge_width + line_offset, 0),
        (left_edge_width + line_offset, frame_ht),
        line_color,
        line_thickness,
    )  # Left line

    cv2.line(
        bg_pattern_horizontal,
        (right_edge_start - line_offset, 0),
        (right_edge_start - line_offset, frame_ht),
        line_color,
        line_thickness,
    )  # Right line

    cv2.line(
        bg_pattern_vertical,
        (0, top_bar_height + line_offset),
        (frame_wd, top_bar_height + line_offset),
        line_color,
        line_thickness,
    )  # Top line

    cv2.line(
        bg_pattern_vertical,
        (0, bottom_bar_start - line_offset),
        (frame_wd, bottom_bar_start - line_offset),
        line_color,
        line_thickness,
    )  # Bottom line

    # Precompute the mask to black out corners
    mask = np.ones((frame_ht, frame_wd, 3), dtype=np.uint8) * 255
    corner_extension = 2 * line_thickness  # Extend the mask to cover the corners
    # Top-left corner
    cv2.rectangle(
        mask,
        (0, 0),
        (left_edge_width + corner_extension, top_bar_height + corner_extension),
        (0, 0, 0),
        -1,
    )

    # Top-right corner
    cv2.rectangle(
        mask,
        (right_edge_start - corner_extension, 0),
        (frame_wd, top_bar_height + corner_extension),
        (0, 0, 0),
        -1,
    )

    # Bottom-left corner
    cv2.rectangle(
        mask,
        (0, bottom_bar_start - corner_extension),
        (left_edge_width + corner_extension, frame_ht),
        (0, 0, 0),
        -1,
    )

    # Bottom-right corner
    cv2.rectangle(
        mask,
        (right_edge_start - corner_extension, bottom_bar_start - corner_extension),
        (frame_wd, frame_ht),
        (0, 0, 0),
        -1,
    )

    # Initialize background offsets
    bg_offset_horizontal = 0  # Offset for horizontal motion
    bg_offset_vertical = 0  # Offset for vertical motion

    for i in tqdm(range(frame_num)):
        # Read frames
        ret_trans, frame_trans = input_trans_cap.read()
        ret_ftir, frame_ftir = input_ftir_cap.read()
        if not ret_trans or not ret_ftir:
            print(f"Error: Could not read frame {i}")
            break

        # Transform frames
        frame_trans_transformed = four_point_transform(
            frame_trans[:, :, 0], orientation_vector[i], center[i], frame_wd, frame_ht
        )
        frame_trans_transformed = np.stack([frame_trans_transformed] * 3, axis=-1)

        frame_ftir_transformed = four_point_transform(
            frame_ftir[:, :, 0], orientation_vector[i], center[i], frame_wd, frame_ht
        )
        frame_ftir_transformed = np.stack([frame_ftir_transformed] * 3, axis=-1)

        # Update background offsets
        bg_offset_horizontal = (
            bg_offset_horizontal - int(speed_along_orientation[i])
        ) % frame_ht
        bg_offset_vertical = (
            bg_offset_vertical + int(perpendicular_velocity[i])
        ) % frame_wd

        # Create moving backgrounds
        moving_bg_horizontal = np.roll(
            bg_pattern_horizontal, -bg_offset_horizontal, axis=0
        )  # Roll vertically
        moving_bg_vertical = np.roll(
            bg_pattern_vertical, -bg_offset_vertical, axis=1
        )  # Roll horizontally

        # Combine both backgrounds
        combined_bg = cv2.addWeighted(moving_bg_horizontal, 1, moving_bg_vertical, 1, 0)

        # Apply the mask to black out corners
        combined_bg = cv2.bitwise_and(combined_bg, mask)

        # Overlay the combined background with the transformed frames
        blended_frame_trans = cv2.addWeighted(
            combined_bg, 0.5, frame_trans_transformed, 1, 0
        )
        blended_frame_ftir = cv2.addWeighted(
            combined_bg, 0.5, frame_ftir_transformed, 1, 0
        )

        # Write blended frames to the output videos
        output_trans.write(blended_frame_trans)
        output_ftir.write(blended_frame_ftir)

    # Release resources
    input_trans_cap.release()
    input_ftir_cap.release()
    output_trans.release()
    output_ftir.release()

    # print(f"Processed videos saved to {output_trans_path} and {output_ftir_path}")


def process_recording(recording):
    print(f"Processing {os.path.basename(recording)}...")

    recording_name = os.path.basename(recording)
    trans_path = os.path.join(recording, "trans_resize.avi")
    ftir_path = os.path.join(recording, "ftir_resize.avi")

    dlc_postfix = "DLC_resnet50_arcteryx500Nov4shuffle1_350000"
    # dlc_path = os.path.join(recording, "trans_resize" + dlc_postfix + ".h5")

    dlc_path = os.path.join(recording, "trans_resize" + dlc_postfix + "_filtered.h5")
    dest_path = os.path.join(recording, "features.h5")

    extract_features(recording_name, ftir_path, dlc_path, dest_path)
    extract_posture_with_moving_background(
        recording_name, trans_path, ftir_path, dlc_path
    )
