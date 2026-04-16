from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import numpy as np

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
        fps = f[animal_name]['fps'][()]

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

    centroid = cal_centroid(label)

    # trim the time series by animal detection
    centroid = centroid[start_frame:]

    # for now, normalize x,y location to [0,1] by the frame size
    centroid = centroid / frame_size

    # clean the data to remove NaNs before plotting
    clean_data = pd.DataFrame({"x": centroid["x"] * scale, "y": centroid["y"] * scale}).dropna()

    # flip the y-axis to match the video recording
    clean_data["y"] = 1.0 - clean_data["y"]

    # take the centroid x,y location tracking and generate a heatmap for the occupancy
    step = max(1, int(fps))
    kde_data = clean_data.iloc[::step].copy()
    jitter_x = kde_data["x"] + np.random.normal(0, 1e-5, len(kde_data["x"]))
    jitter_y = kde_data["y"] + np.random.normal(0, 1e-5, len(kde_data["y"]))

    fig, ax = plt.subplots(figsize=[8, 8])

    # Set the inner plot area AND the outer figure border to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # KDE heatmap
    try:
        sns.kdeplot(
            x=jitter_x,
            y=jitter_y,
            fill=True,
            cmap="inferno",
            thresh=0.02,
            levels=100,
            clip=((0, 1), (0, 1)),
            bw_adjust=1.2,  # Smooths the data slightly to ensure contour levels can be calculated
            ax=ax
        )
    except ValueError as e:
        # 3. The Ultimate Fallback: If 100 levels STILL fails on a weird edge case,
        # drop the complexity to 10 levels so the pipeline doesn't crash the deployment.
        print(f"Warning: KDE 100-level failed ({e}). Falling back to low-res KDE.")
        sns.kdeplot(
            x=jitter_x,
            y=jitter_y,
            fill=True,
            cmap="inferno",
            thresh=0.05,
            levels=10,
            clip=((0, 1), (0, 1)),
            bw_adjust=2.0,
            ax=ax
        )

    # Overlay trajectory
    ax.plot(clean_data["x"], clean_data["y"], color="white", alpha=0.3, lw=0.5)

    # Labels, ticks, and formatting applied to the axes object
    ax.set_title("Open Field Occupancy Map", color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Normalized X Position", color="white")
    ax.set_ylabel("Normalized Y Position", color="white")
    ax.grid(False)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()

    plt.savefig(dest_path, bbox_inches="tight", pad_inches=0, dpi=600, facecolor=fig.get_facecolor())
    plt.close()
    print(f"occupancy map saved to {dest_path}")


def cal_centroid(label):
    """
    input:
    label: DLC tracking of the recording
    return: the x,y location of the (estimated) centroid of the mouse
    """

    # for now hard-code the list of bodyparts used to estimate centroid
    bp_list = [
        # 'tailbase',
        'hip',
        'sternumtail',
        'sternumhead',
        'neck',
        # 'snout',
        'lhip',
        'rhip',
        'lshoulder',
        'rshoulder'
    ]

    sub = label.loc[:, pd.IndexSlice[bp_list, ['x', 'y', 'likelihood']]]
    x = sub.xs('x', axis=1, level=-1)
    y = sub.xs('y', axis=1, level=-1)
    lik = sub.xs('likelihood', axis=1, level=-1)

    # likelihood-weighted sum that integrate different bodyparts locations
    weighted_x = (x * lik).sum(axis=1)
    weighted_y = (y * lik).sum(axis=1)
    sum_weights = lik.sum(axis=1)
    mean_x = weighted_x.div(sum_weights).where(sum_weights.ne(0))
    mean_y = weighted_y.div(sum_weights).where(sum_weights.ne(0))

    centroid = pd.concat([mean_x, mean_y], axis=1)
    centroid.columns = ['x', 'y']

    return centroid


def cal_displacement(
        features_h5: str,
        tracking_h5: str,
        window_sec: float = 0.5,
        smooth_sigma: int = 3
):
    """
    Calculate the displacement of the animal relative to a rolling mean location
    in the past N seconds, and save it to the same features.h5 file.

    Parameters
    ----------
    features_h5: full path of the features_h5 file
    tracking_h5: full path of the tracking_h5 file
    window_sec : float, optional
        Size of rolling window in seconds (default = 0.5).
    smooth_sigma : int, optional
        Gaussian smoothing sigma in frames to reduce jitter (default = 3).

    Returns
    -------
    displacement_px : np.ndarray
        Displacement (in pixels) from rolling mean location.
    """

    # --- basic setup ---
    with h5py.File(features_h5, 'r') as f:
        group_names = list(f.keys())
        if not group_names:
            raise ValueError(f"No group names found in {features_h5}")

        animal_name = group_names[0]
        frame_size = f[animal_name]['frame_size'][()]
        frame_count = f[animal_name]['frame_count'][()]
        fps = f[animal_name]['fps'][()]

    half_window = int((window_sec * fps) / 2)

    df = pd.read_hdf(tracking_h5)
    model_id = df.columns[0][0]
    label = df[model_id]

    centroid = cal_centroid(label)
    x = centroid["x"]
    y = centroid["y"]

    # --- smooth position ---
    x_smooth = gaussian_filter1d(x, sigma=smooth_sigma)
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)

    n = len(x_smooth)
    displacement_px = np.zeros(n)

    # --- compute centered displacement ---
    for i in range(n):
        left = max(i - half_window, 0)
        right = min(i + half_window, n - 1)
        dx = x_smooth[right] - x_smooth[left]
        dy = y_smooth[right] - y_smooth[left]
        displacement_px[i] = np.hypot(dx, dy)

    # --- write back to the same features file ---
    with h5py.File(features_h5, 'a') as f:
        g = f[animal_name]
        if "displacement_px" in g.keys():
            del g["displacement_px"]
        g.create_dataset("displacement_px", data=displacement_px)

    print(f"displacement_px saved to group '{animal_name}' in {features_h5}")

    return displacement_px