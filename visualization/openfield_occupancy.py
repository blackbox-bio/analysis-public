from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import h5py


def plot_open_field_occupancy_map(
        features_h5: str,
        tracking_h5: str,
        dest_path: str):

    '''
    :param features_h5: full path of the features_h5 file
    :param tracking_h5: full path of the tracking_h5 file
    :param dest_path: full path of the output plot file, saved as PNG right now

    The function takes the tailbase tracking and plot a heatmap to visualize
    the occupancy map of the animal in a given recording
    '''

    # read features.h5 file
    with h5py.File(features_h5, 'r') as f:
        group_names = list(f.keys())
        if not group_names:
            raise ValueError(f"No group names found in {features_h5}")

        animal_name = group_names[0]
        frame_size = f[animal_name]['frame_size'][()]
        frame_count = f[animal_name]['frame_count'][()]

        # use animal detection to dynamically trim the beginning of the recording with an empty field
        start_frame = 0
        if "animal_detection" in f[animal_name].keys():
            animal_detection = f[animal_name]['animal_detection'][:]

            for i in range(frame_count):
                if animal_detection[i] == 1:
                    start_frame = i
                    break

        # field_size = f[animal_name['field_size'][()]
        # TODO: Once feature could read the physical size of the recording field,
        # we could convert the x,y axis to actual distance unit such as cm
        # scale = field_size / frame_size  # 2x2 mouse or openfield would be scale = 30/1024
        scale = 1.0  # for now, scale set to 1

    # read tracking.h5 file
    df = pd.read_hdf(tracking_h5)
    model_id = df.columns[0][0]
    label = df[model_id]
    tailbase = label['tailbase'][['x','y']][start_frame:frame_count]

    # for now, normalize x,y location to [0,1] by the frame size
    tailbase = tailbase / frame_size

    # take the tailbase x,y location tracking and generate a heatmap for the occupancy
    plt.figure(figsize=[8, 8])

    # KDE heatmap
    sns.kdeplot(
        x=tailbase["x"] * scale,
        y=tailbase["y"] * scale,
        fill=True, cmap="inferno",
        thresh=0, levels=100
    )
    # Overlay trajectory
    plt.plot(tailbase["x"] * scale, tailbase["y"] * scale, color="white", alpha=0.3, lw=0.5)

    # Flip y-axis
    plt.gca().invert_yaxis()

    # Labels and ticks
    plt.title("Open Field Occupancy Map")
    # plt.xlabel("X position")
    # plt.ylabel("Y position")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(dest_path, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close()
    print(f"occupancy map saved to {dest_path}")

