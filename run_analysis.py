from process import *
from summary import *
from dlc_runner import *
import concurrent.futures


def main():

    # Ask the user to select subfolder to process
    root = tk.Tk()
    root.withdraw()

    # input the path to the experiment folder
    exp_folder = select_folder()

    # run dlc
    body_videos = get_body_videos([exp_folder])
    run_deeplabcut(dlc_config_path, body_videos)

    features_folder = os.path.join(exp_folder, "features")
    if not os.path.exists(features_folder):
        # Create the directory
        os.makedirs(features_folder)

    # generate the list of videos to be processed
    video_list = []
    for i in os.listdir(os.path.join(exp_folder, "videos")):
        if "body" in i and ".avi" in i:
            video_list.append(i.split("_body")[0])

    print(f"In total {len(video_list)} videos to be processed: \n{video_list}")

    # # Process the videos iteratively
    # for video in video_list:
    #     process_video(video, exp_folder, features_folder)

    # Get the number of available CPU cores
    num_workers = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    # Create a list of argument tuples for process_video
    video_args_list = [(video, exp_folder, features_folder) for video in video_list]

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map the process_video_wrapper function to the list of arguments
        executor.map(process_video_wrapper, video_args_list)

    # generate summary csv from the processed videos
    summary_csv = os.path.join(exp_folder, "summary.csv")
    generate_summary_csv(features_folder, summary_csv)


if __name__ == "__main__":
    main()
