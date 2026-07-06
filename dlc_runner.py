from palmreader_analysis.events import PalmreaderProgress
import os
import deeplabcut
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# info = os.uname()
#
# if os.name == "nt":
#     dlc_config_path = r"D:\DLC\blackbox_dlc_deployment\config.yaml"
# if os.name == "posix":
#     dlc_config_path = r"/Users/zihealexzhang/work_local/blackbox_data/arcteryx500-alex-2023-11-04/config.yaml"
# if info.sysname == "Linux":
#     print("Running on Linux, right now dedicated to torch backend")
#     dlc_config_path = r"/home/alex/Documents/DLC/dlc-torch-deployment/config.yaml"


# Function to run DeepLabCut on the specified videos
#
# THIS IS AN API ENTRYPOINT! If the signature is modified, ensure api.py matches!
# The body of this function can change without affecting the API.
def run_deeplabcut(
    dlc_config_path,
    body_videos,
    generate_skeleton_flag=True,
    simple_skeleton_flag=False,
    gpu_idx=0,
):
    PalmreaderProgress.start_multi(
        len(body_videos), "Analyzing videos", autoincrement=True
    )

    deeplabcut.analyze_videos(
        dlc_config_path,
        body_videos,
        videotype=".mp4",
        shuffle=0,
        device=f"cuda:{gpu_idx}",
    )

    PalmreaderProgress.start_multi(len(body_videos), "Filtering predictions")

    for video in body_videos:
        PalmreaderProgress.increment_multi()

        deeplabcut.filterpredictions(dlc_config_path, [video], shuffle=0, save_as_csv=False)
        # deeplabcut.create_labeled_video(
        #     dlc_config_path, [video], videotype=".avi", filtered=True
        # )

    if generate_skeleton_flag:
        generate_skeleton(dlc_config_path, body_videos, simple_skeleton_flag)

    return


# Function to generate a skeleton video from the specified videos
#
# THIS IS AN API ENTRYPOINT! If the signature is modified, ensure api.py matches!
# The body of this function can change without affecting the API.
def generate_skeleton(dlc_config_path, body_videos, simple_skeleton_flag=False):
    PalmreaderProgress.start_single("Generating skeleton videos", parallel=True)

    bodyparts = [
        # "tailtip",
        "tailbase",
        "hip",
        "sternumtail",
        "sternumhead",
        "neck",
        "snout",
        "lhip",
        "rhip",
        "lshoulder",
        "lankle",
        "rankle",
        "rshoulder",
        "lhpaw",
        "rhpaw",
        "lfpaw",
        "rfpaw"
    ]

    if simple_skeleton_flag:
        for video in body_videos:
            deeplabcut.create_labeled_video(
                dlc_config_path,
                [video],
                shuffle=0,
                displayedbodyparts=bodyparts,
                filtered=True,
                draw_skeleton=True,
                overwrite=True,
            )
    else:
        # generate full skeleton
        for video in body_videos:
            deeplabcut.create_labeled_video(
                dlc_config_path,
                [video],
                shuffle=0,
                filtered=True,
                draw_skeleton=True,
                overwrite=True,
            )

    return
