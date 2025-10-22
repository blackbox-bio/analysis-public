from matplotlib import pyplot as plt
import seaborn as sns


def plot_open_field_occupancy_map(self, recording_name):
    # TODO: implement a scale convertor by taking recording metadata
    #  (2x2 mouse, openfield mouse, or openfield rat)
    #  for converting the x and y axis labels from pixel to cm.
    scale = 30 / 1024  # 2x2 mouse or openfield

    field_size = 30  # 30x30 cm arena
    # frame_size = 1024 # 1024x1024 frame resolution for the recording
    # scale = field_size/frame_size

    # take the tailbase x,y location tracking and generate a heatmap for the occupancy
    tailbase = self.label['tailbase'][['x', 'y']][:]
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
    plt.xlabel("X position (cm)")
    plt.ylabel("Y position (cm)")
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    plt.tight_layout()

    png = os.path.join(recording_name, "openfield_occupancy_map.png")
    plt.savefig(png, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close()
    print(f"occupancy map saved to {png}")