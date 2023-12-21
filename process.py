from utils import *

dlc_postfix = "DLC_resnet50_arcteryx500Nov4shuffle1_350000"

# Function to process a video with specified arguments
def process_video_wrapper(video, exp_folder, features_folder):
    return process_video(video, exp_folder, features_folder)


def process_video(video, exp_folder, output_folder):

    print(f"Processing {video}...")

    dlc_path = os.path.join(
        exp_folder, "videos", video + "_body" + dlc_postfix + "_filtered.h5"
    )

    # create a dictionary to store the extracted features
    features = {}

    # read DLC tracking
    df = pd.read_hdf(dlc_path)
    label = df[dlc_postfix]

    # calculate distance traveled
    features["distance_traveled"] = np.nansum(cal_distance_(label)).reshape(-1, 1)

    # ----calculate paw luminance, average paw luminance ratio, and paw luminance log-ratio----
    # read ftir video
    ftir_video = cv2.VideoCapture(
        os.path.join(exp_folder, "videos", f"{video}_ftir.avi")
    )
    fps = int(ftir_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(ftir_video.get(cv2.CAP_PROP_FRAME_COUNT))
    recording_time = frame_count / fps

    features["recording_time"] = np.array(recording_time)
    # calculate paw luminance
    (
        hind_left,
        hind_right,
        front_left,
        front_right,
        background_luminance,
    ) = cal_paw_luminance(label, ftir_video, size=22)

    features["hind_left_luminance"] = hind_left
    features["hind_right_luminance"] = hind_right
    features["front_left_luminance"] = front_left
    features["front_right_luminance"] = front_right
    features["background_luminance"] = background_luminance

    hind_left_scaled, hind_right_scaled = scale_ftir(hind_left, hind_right)
    features["hind_left_luminance_scaled"] = hind_left_scaled
    features["hind_right_luminance_scaled"] = hind_right_scaled

    # calculate luminance logratio
    features["average_luminance_ratio"] = np.nansum(
        features["hind_left_luminance"]
    ) / np.nansum(features["hind_right_luminance"]).reshape(-1, 1)
    features["luminance_logratio"] = np.log(
        (features["hind_left_luminance_scaled"] + 1e-4)
        / (features["hind_right_luminance_scaled"] + 1e-4)
    )

    # calculate when the animal is standing on two hind paws
    features["standing_on_two_paws"] = cal_stand_on_two_paws(front_left, front_right)
    # -------------------------------------------------------------

    # save extracted features
    with h5py.File(os.path.join(output_folder, f"{video}.h5"), "w") as hdf:
        video_data = hdf.create_group(video)
        for key in features.keys():
            video_data.create_dataset(key, data=features[key])
