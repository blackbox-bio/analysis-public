from utils import *
from palmreader_analysis import FeaturesContext


# Function to process a video with specified arguments
def process_recording_wrapper(recording):
    return process_recording(recording)


# Extract features from a video
#
# THIS IS AN API ENTRYPOINT! If the signature is modified, ensure api.py matches!
# The body of this function can change without affecting the API.
def extract_features(name, ftir_path, tracking_path, dest_path):
    ctx = FeaturesContext(name, tracking_path, ftir_path)

    for feature in FeaturesContext.get_all_features():
        feature.extract(ctx=ctx)

    ctx.to_hdf5(dest_path)


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
    # extract_posture_with_moving_background(recording_name, trans_path, ftir_path, dlc_path)
