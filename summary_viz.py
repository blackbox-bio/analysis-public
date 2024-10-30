from io import BytesIO
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import cv2 as cv
from enum import Enum
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore


class GraphType(Enum):
    KDE = "kde"
    HIST = "hist"
    REG = "reg"

    def __str__(self):
        return self.value


def get_plot_fn(kind: str):
    _kind = GraphType(kind)

    if _kind == GraphType.KDE:
        return sns.kdeplot
    elif _kind == GraphType.HIST:
        return sns.histplot
    elif _kind == GraphType.REG:
        return sns.regplot
    else:
        raise ValueError(f"Unknown graph type: {_kind}")


def summary_viz_preprocess(df, rows_to_include, columns_to_include, group_variable):
    """
    preprocess the summary csv file for visualization with user inputs
    df: the summary.csv file
    rows_to_include: a boolean vector mask for selecting a subset of the recordings
    columns_to_include: a list of columns to be included in the plot
    group_variable: the grouping variable for the plot hue

    df_plot: the processed df for visualization
    """
    columns_to_include.append(group_variable)
    df_plot = df[columns_to_include]
    df_plot = df_plot[rows_to_include]

    return df_plot


def rank_columns_by_significance(df, group_variable):
    # Dictionary to store p-values for each numerical column
    p_values = {}

    # Group DataFrame by the specified column
    grouped = df.groupby(group_variable)

    # Iterate over numerical columns
    for column in df.select_dtypes(include="number"):
        # List to store p-values for each column
        column_p_values = []

        # Iterate over groups and conduct statistical test
        for group_name, group_data in grouped:
            other_groups_data = grouped.get_group(group_name)[column]
            p_value = (
                ttest_ind(group_data[column], other_groups_data)[1]
                if len(grouped) == 2
                else f_oneway(group_data[column], other_groups_data)[1]
            )
            column_p_values.append(p_value)

        # Store the maximum p-value as the significance measure
        p_values[column] = max(column_p_values)

    # Sort columns by significance (ascending order)
    sorted_columns = sorted(p_values, key=p_values.get)

    return sorted_columns


def _plot_to_cv2_image(plot):
    buf = BytesIO()
    plot.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    img = cv.imdecode(np.frombuffer(buf.read(), np.uint8), cv.IMREAD_UNCHANGED)
    return img


def _horizontal_concat_step(prev, next):
    """
    Concatenate two images horizontally. If `prev` is None, return `next`.
    """
    if prev is None:
        return next

    return cv.hconcat([prev, next])


def _get_group_label(group_variable):
    """
    Variables provided through Palmreader start with "PV:", which is not necessary for the plot.
    """
    return group_variable[4:] if group_variable.startswith("PV:") else group_variable


def generate_bar_plots(df, group_variable: str, dest_path, sort_by_significance=False):
    """
    Generate a bar plot for each column in `df` grouped by `group_variable`.

    This function will independently generate a graph for each column then concatenate them horizontally,
    saving the result to `dest_path`.

    If `sort_by_significance` is True, the columns will be ranked by significance between groups.
    """

    # Rank columns by significance
    if sort_by_significance:
        sorted_columns = rank_columns_by_significance(df, group_variable)
    else:
        sorted_columns = df.select_dtypes(include="number").columns

    joined = None

    group_label = _get_group_label(group_variable)

    # Generate individual bar plots for each numerical column
    for column in sorted_columns:
        plt.figure()

        # sns.boxplot(
        #     x=group_variable,
        #     y=column,
        #     data=df,
        #     # fill = False,
        #     # split = True,
        #     # legend = 'auto',
        #     # dodge=True,
        # )
        # ax = sns.barplot(
        #     x=group_variable,
        #     y=column,
        #     data=df,
        #     edgecolor="black",
        #     linewidth=2,
        #     fill=False,
        #     errorbar="se",
        #     capsize=0.1,
        # )
        ax = sns.pointplot(
            x=group_variable,
            y=column,
            data=df,
            hue=group_variable,
            # markers="_",
            # scale=0.5,
            # markersize=10,
            errorbar="se",
            capsize=0.1,
        )

        sns.stripplot(
            x=group_variable,
            y=column,
            data=df,
            # color="black",
            hue=group_variable,
            # dodge=True,
            alpha=0.4,
            jitter=True,
            legend=False,
            ax=ax,
        )

        ax.set_facecolor("none")
        # Remove the legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        plt.title(f"{column} grouped by {group_label}", wrap=True)

        plot = plt.gcf()
        # remove PV: from the x-axis label
        plot.get_axes()[0].set_xlabel(group_label)

        img = _plot_to_cv2_image(plot)
        joined = _horizontal_concat_step(joined, img)

    cv.imwrite(dest_path, joined)


def generate_PairGrid_plot(
    df,
    group_variable: str,
    diag_kind: str,
    upper_kind: str,
    lower_kind: str,
    dest_path: str,
    sort_by_significance=False,
):

    # Rank columns by significance
    if sort_by_significance:
        sorted_columns = rank_columns_by_significance(df, group_variable)
    else:
        sorted_columns = df.select_dtypes(include="number").columns

    group_label = _get_group_label(group_variable)

    g = sns.PairGrid(df, hue=group_variable, diag_sharey=False)

    diag = get_plot_fn(diag_kind)
    upper = get_plot_fn(upper_kind)
    lower = get_plot_fn(lower_kind)

    g.map_diag(diag)
    g.map_upper(upper)
    g.map_lower(lower)
    g.add_legend(adjust_subtitles=True, title=group_label)

    # rotate all axis labels
    for arr in g.axes:
        for axis in arr:
            axis.set_xlabel(axis.get_xlabel(), rotation=10)
            axis.set_ylabel(axis.get_ylabel(), rotation=80, labelpad=25)

    g.savefig(dest_path, dpi=300)


def generate_heatmap_plot(df, group_variable: str, dest_path: str):
    """
    Generate a heatmap of group-level mean Z-scores with significance markers.

    Parameters:
    df : pd.DataFrame
        The dataframe containing the features and group labels.
    group_variable : str
        The column name of the group variable (treatment group).
    dest_path : str
        The file path where the heatmap plot will be saved.

    """
    # group_label = _get_group_label(group_variable)

    # Step 1: Apply Z-score normalization to each feature column (excluding the group variable)
    feature_cols = df.columns.drop(
        group_variable
    )  # Exclude the group variable from the feature columns
    df_zscore = df[feature_cols].apply(
        zscore, axis=0
    )  # Z-score normalization for features

    # Step 2: Add the group variable column back to the dataframe
    df_zscore[group_variable] = df[group_variable]

    # Step 3: Calculate mean Z-scores for each group
    df_mean_zscore = df_zscore.groupby(group_variable).mean()

    # Step 4: Run ANOVA for each feature to find significant differences between groups
    p_values = []
    for feature in df_mean_zscore.columns:
        groups = [
            df_zscore[df_zscore[group_variable] == group][feature]
            for group in df_zscore[group_variable].unique()
        ]
        stat, p = f_oneway(*groups)  # Perform ANOVA
        p_values.append(p)

    # Step 5: Adjust p-values using Bonferroni or FDR correction
    p_adjusted = multipletests(p_values, method="bonferroni")[
        1
    ]  # Bonferroni correction (you can also use 'fdr_bh')

    # Step 6: Create a significance marker array
    significance_array = np.full(
        df_mean_zscore.T.shape, "", dtype=object
    )  # Empty array for significance markers

    # Assign significance markers based on adjusted p-values
    for i, p in enumerate(p_adjusted):
        if p < 0.001:
            significance_array[i] = "***"
        elif p < 0.01:
            significance_array[i] = "**"
        elif p < 0.05:
            significance_array[i] = "*"

    # Step 7: Create a clustermap with significance markers on top (no need to transpose before this step)
    g = sns.clustermap(
        df_mean_zscore.T,
        cmap="inferno",
        center=0,
        annot=significance_array,
        fmt="",
        cbar_kws={"label": "Mean Z-score"},
        metric="euclidean",
        method="ward",
    )

    # extract the order of the features after clustering
    row_order = g.dendrogram_row.reordered_ind

    # Step 8: Force all y-axis labels (feature names) to be displayed
    g.ax_heatmap.set_yticks(
        np.arange(len(row_order)) + 0.5
    )  # Set y-ticks for every row
    g.ax_heatmap.set_yticklabels(
        df_mean_zscore.columns[row_order], rotation=0, fontsize=10
    )  # Apply reordered labels

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right"
    )

    # Step 9: Customize the title and display the plot
    plt.title(
        "Group-Level Mean Z-scores with Significance Markers and All Feature Labels"
    )

    # Step 10: Save the plot to the specified destination path
    plt.savefig(dest_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display in environments that render plots automatically

    return
