import os
import tkinter as tk
from tkinter import filedialog
import deeplabcut


selected_folders = []


# Function to get the paths to the body videos within the specified folders
# def get_body_videos(folder_paths):
#     body_videos = []
#     for folder_path in folder_paths:
#         video_subfolder_path = os.path.join(folder_path, 'videos')
#         if os.path.exists(video_subfolder_path):
#             videos = [os.path.join(video_subfolder_path, f) for f in os.listdir(video_subfolder_path) if
#                       f.endswith('_body.avi')]
#             body_videos.extend(videos)
#     return body_videos

def get_body_videos(directorys):
    # Get the current time
    # current_time = datetime.now()
    # Calculate the time 24 hours ago
    # twenty_four_hours_ago = current_time - timedelta(hours=24)

    avi_files = []

    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file in files:
                # file_path = os.path.join(root, file)
                if file.endswith(".avi") and "_body" in file:
                    # Check if the file is created in the last 24 hours
                    # if os.path.getctime(file_path) > twenty_four_hours_ago.timestamp():
                    avi_files.append(os.path.join(root, file))
    return avi_files

# Function to run DeepLabCut on the specified videos
def run_deeplabcut(dlc_config_path, body_videos):
    deeplabcut.analyze_videos(dlc_config_path, body_videos, videotype='.avi')
    deeplabcut.filterpredictions(dlc_config_path, body_videos, save_as_csv=False)

    return

# Function to prompt the user to select folders using a GUI dialog
def select_folders():
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select subfolders to process
    selected_folders = []
    while True:
        folder = filedialog.askdirectory(parent=root, title='Select a subfolder to process (Cancel to finish)')
        if folder:
            selected_folders.append(folder)
        else:
            break

    return selected_folders


def main():
    root = tk.Tk()
    root.withdraw()

    dlc_config_path = r'D:\DLC\arcteryx500-alex-2023-11-04\config.yaml'

    # Ask the user to select subfolders to process
    selected_folders = select_folders()

    # Get all '_body' videos in the selected subfolders
    body_videos = get_body_videos(selected_folders)
    print("Found body_videos:", body_videos)

    # Run DeepLabCut analyze_videos function on the body videos
    run_deeplabcut(dlc_config_path, body_videos)

    print(f"Found these folders: {selected_folders}")

    return


if __name__ == '__main__':
    main()
