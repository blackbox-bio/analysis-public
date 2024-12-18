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


# def generate_heatmap_plot(df, group_variable: str, dest_path: str):
#     """
#     Generate a heatmap of group-level mean Z-scores with significance markers.
#
#     Parameters:
#     df : pd.DataFrame
#         The dataframe containing the features and group labels.
#     group_variable : str
#         The column name of the group variable (treatment group).
#     dest_path : str
#         The file path where the heatmap plot will be saved.
#
#     """
#     # group_label = _get_group_label(group_variable)
#
#     # Step 1: Apply Z-score normalization to each feature column (excluding the group variable)
#     feature_cols = df.columns.drop(
#         group_variable
#     )  # Exclude the group variable from the feature columns
#     df_zscore = df[feature_cols].apply(
#         zscore, axis=0
#     )  # Z-score normalization for features
#
#     # Step 2: Add the group variable column back to the dataframe
#     df_zscore[group_variable] = df[group_variable]
#
#     # Step 3: Calculate mean Z-scores for each group
#     df_mean_zscore = df_zscore.groupby(group_variable).mean()
#
#     # Step 4: Run ANOVA for each feature to find significant differences between groups
#     p_values = []
#     for feature in df_mean_zscore.columns:
#         groups = [
#             df_zscore[df_zscore[group_variable] == group][feature]
#             for group in df_zscore[group_variable].unique()
#         ]
#         stat, p = f_oneway(*groups)  # Perform ANOVA
#         p_values.append(p)
#
#     # Step 5: Adjust p-values using Bonferroni or FDR correction
#     p_adjusted = multipletests(p_values, method="bonferroni")[
#         1
#     ]  # Bonferroni correction (you can also use 'fdr_bh')
#
#     # Step 6: Create a significance marker array
#     significance_array = np.full(
#         df_mean_zscore.T.shape, "", dtype=object
#     )  # Empty array for significance markers
#
#     # Assign significance markers based on adjusted p-values
#     for i, p in enumerate(p_adjusted):
#         if p < 0.001:
#             significance_array[i] = "***"
#         elif p < 0.01:
#             significance_array[i] = "**"
#         elif p < 0.05:
#             significance_array[i] = "*"
#
#     # Step 7: Create a clustermap with significance markers on top (no need to transpose before this step)
#     g = sns.clustermap(
#         df_mean_zscore.T,
#         cmap="inferno",
#         center=0,
#         annot=significance_array,
#         fmt="",
#         cbar_kws={"label": "Mean Z-score"},
#         metric="euclidean",
#         method="ward",
#     )
#
#     # extract the order of the features after clustering
#     row_order = g.dendrogram_row.reordered_ind
#
#     # Step 8: Force all y-axis labels (feature names) to be displayed
#     g.ax_heatmap.set_yticks(
#         np.arange(len(row_order)) + 0.5
#     )  # Set y-ticks for every row
#     g.ax_heatmap.set_yticklabels(
#         df_mean_zscore.columns[row_order], rotation=0, fontsize=10
#     )  # Apply reordered labels
#
#     g.ax_heatmap.set_xticklabels(
#         g.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right"
#     )
#
#     # Step 9: Customize the title and display the plot
#     plt.title(
#         "Group-Level Mean Z-scores with Significance Markers and All Feature Labels"
#     )
#
#     # Step 10: Save the plot to the specified destination path
#     plt.savefig(dest_path, dpi=300, bbox_inches="tight")
#     plt.close()  # Close the plot to avoid display in environments that render plots automatically
#
#     return


def plot_individual_heatmap_by_group(
        df_individual, group_variable, dest_path, group_colors, ordered_features, lut
):
    """
    Plot individual heatmap, sorted by group without clustering.

    Parameters:
    df_individual : pd.DataFrame
        Z-score normalized dataframe of individual records.
    group_variable : str
        The column name of the group variable (e.g., treatment group).
    dest_path : str
        The file path to save the heatmap plot.
    group_colors : pd.Series
        Series containing the group colors for each individual.
    ordered_features : list
        List of ordered feature names based on group-level clustering.
    lut : dict
        Lookup table mapping group labels to colors.
    """
    unique_groups = df_individual[group_variable].unique()
    # Sort individual records by group
    df_individual.sort_values(by=group_variable, inplace=True)
    group_colors = df_individual[group_variable].map(
        lut
    )  # Adjust color mapping after sorting
    df_individual.drop(columns=[group_variable], inplace=True)  # Drop the group column

    # Use clustermap with no clustering (row_cluster=False, col_cluster=False)
    g_ind = sns.clustermap(
        df_individual.T,
        cmap="inferno",
        center=0,
        col_colors=group_colors,  # Add the group color row at the top
        row_cluster=False,  # Disable clustering for rows
        col_cluster=False,  # Disable clustering for columns
        cbar_kws={"label": "Z-score"},
    )

    # Force all y-axis labels (feature names) to be displayed
    g_ind.ax_heatmap.set_yticks(
        np.arange(len(ordered_features)) + 0.5
    )  # Set y-ticks for every row
    g_ind.ax_heatmap.set_yticklabels(
        ordered_features, rotation=0, fontsize=10
    )  # Apply reordered labels

    g_ind.ax_heatmap.set_xticklabels(
        g_ind.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right"
    )

    # Add legend for the group colors
    for label in unique_groups:
        g_ind.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    g_ind.ax_col_dendrogram.legend(
        title=group_variable, loc="center", ncol=len(unique_groups)
    )

    # Customize the title and save the plot
    plt.title(f"Individual Z-scores (Sorted by Group)")
    plt.savefig(dest_path, dpi=300, bbox_inches="tight")
    plt.close(g_ind.fig)  # Close the clustermap figure

    return


# Helper function to plot individual heatmap by clustering
def plot_individual_heatmap_by_clustering(
        df_individual, group_variable, dest_path, group_colors, ordered_features, lut
):
    """
    Plot individual heatmap, sorted by clustering.

    Parameters:
    df_individual : pd.DataFrame
        Z-score normalized dataframe of individual records.
    group_variable : str
        The column name of the group variable (e.g., treatment group).
    dest_path : str
        The file path to save the heatmap plot.
    group_colors : pd.Series
        Series containing the group colors for each individual.
    ordered_features : list
        List of ordered feature names based on group-level clustering.
    lut : dict
        Lookup table mapping group labels to colors.
    """
    #
    unique_groups = df_individual[group_variable].unique()
    df_individual.drop(columns=[group_variable], inplace=True)

    # Plot heatmap with clustering
    g_ind = sns.clustermap(
        df_individual.T,
        cmap="inferno",
        center=0,
        col_colors=group_colors,  # Add the group color row at the top
        metric="euclidean",
        method="ward",  # Perform clustering
    )

    # Force all y-axis labels (feature names) to be displayed
    g_ind.ax_heatmap.set_yticks(
        np.arange(len(ordered_features)) + 0.5
    )  # Set y-ticks for every row
    g_ind.ax_heatmap.set_yticklabels(
        ordered_features, rotation=0, fontsize=10
    )  # Apply reordered labels

    g_ind.ax_heatmap.set_xticklabels(
        g_ind.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right"
    )

    # Add legend for the group colors
    for label in unique_groups:
        g_ind.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    g_ind.ax_col_dendrogram.legend(
        title=group_variable, loc="center", ncol=len(unique_groups)
    )

    # Customize the title and save the plot
    plt.title(f"Individual Z-scores (Sorted by Clustering)")
    plt.savefig(dest_path, dpi=300, bbox_inches="tight")
    plt.close(g_ind.fig)  # Close the clustermap figure

    return


# Master function to handle both group-level and individual-level plots
def generate_heatmap_plots(
        df,
        group_variable: str,
        dest_path_group: str,
        dest_path_individual: str,
        sort_by: str = "clustering",
):
    """
    Master function to generate two heatmaps:
    1. Group-level mean Z-scores with significance markers.
    2. Individual-level Z-scores with the same feature order as the group-level heatmap,
       with an option to sort by group or by clustering.

    Parameters:
    df : pd.DataFrame
        The dataframe containing the features and group labels.
    group_variable : str
        The column name of the group variable (e.g., treatment group).
    dest_path_group : str
        The file path where the group-level heatmap plot will be saved.
    dest_path_individual : str
        The file path where the individual-level heatmap plot will be saved.
    sort_by : str, optional
        Sorting method for individual recordings: "clustering" (default) or "group".
    """
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
    ]  # Bonferroni correction

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

    # Step 7: Create the first heatmap (group-level) with significance markers
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

    # Extract the order of the features after clustering
    row_order = g.dendrogram_row.reordered_ind
    ordered_features = df_mean_zscore.columns[row_order]

    # Force all y-axis labels (feature names) to be displayed
    g.ax_heatmap.set_yticks(
        np.arange(len(row_order)) + 0.5
    )  # Set y-ticks for every row
    g.ax_heatmap.set_yticklabels(
        ordered_features, rotation=0, fontsize=10
    )  # Apply reordered labels

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right"
    )

    # Customize the title and save the plot for group-level heatmap
    plt.title(
        "Group-Level Mean Z-scores with Significance Markers and All Feature Labels"
    )
    plt.savefig(dest_path_group, dpi=300, bbox_inches="tight")
    plt.close(g.fig)  # Ensure the group

    # Step 8: Handle individual plot
    df_individual = df_zscore[
        ordered_features
    ].copy()  # Reorder the features based on the group-level heatmap
    df_individual[group_variable] = df_zscore[
        group_variable
    ]  # Add group variable to the individual dataframe

    # Map groups to colors for the top row
    unique_groups = df_zscore[group_variable].unique()
    lut = dict(
        zip(unique_groups, sns.color_palette("husl", len(unique_groups)))
    )  # Color lookup table
    group_colors = df_zscore[group_variable].map(lut)  # Map group labels to color

    # Call the appropriate function based on the `sort_by` argument
    if sort_by == "group":
        plot_individual_heatmap_by_group(
            df_individual,
            group_variable,
            dest_path_individual,
            group_colors,
            ordered_features,
            lut,
        )
    else:
        plot_individual_heatmap_by_clustering(
            df_individual,
            group_variable,
            dest_path_individual,
            group_colors,
            ordered_features,
            lut,
        )

    return


def generate_cluster_heatmap(
        df,
        group_variable: str,
        dest_path: str,
        grouping_mode: str = "group" # "individual"
):
    """
    Generate a cluster-heatmap plot based on grouping mode.

    Parameters:
    df : pd.DataFrame
        The dataframe containing the features and group labels.
    group_variable : str
        The column name of the group variable (e.g., treatment group).
    dest_path : str
        The file path to save the heatmap plot.
    grouping_mode : str, optional
        Plot type: "group" for group-level cluster-heatmap, "individual" for individual-level heatmap.
    """
    # Step 1: Apply Z-score normalization to each feature column (excluding the group variable)
    feature_cols = df.columns.drop(group_variable)
    df_zscore = df[feature_cols].apply(zscore, axis=0)
    df_zscore[group_variable] = df[group_variable]  # Add back the group variable

    # Step 2: If grouping_mode is "group", create a group-level cluster-heatmap
    if grouping_mode == "group":
        # Calculate mean Z-scores for each group
        df_mean_zscore = df_zscore.groupby(group_variable).mean()

        # Run ANOVA for significance and adjust p-values
        p_values = [
            f_oneway(
                *[df_zscore[df_zscore[group_variable] == group][feature] for group in df_zscore[group_variable].unique()]
            )[1]
            for feature in df_mean_zscore.columns
        ]
        p_adjusted = multipletests(p_values, method="bonferroni")[1]

        # Create a significance marker array
        significance_array = np.full(df_mean_zscore.T.shape, "", dtype=object)
        for i, p in enumerate(p_adjusted):
            if p < 0.001:
                significance_array[i] = "***"
            elif p < 0.01:
                significance_array[i] = "**"
            elif p < 0.05:
                significance_array[i] = "*"

        # Create group-level heatmap
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

        # Extract feature order for subsequent plotting
        row_order = g.dendrogram_row.reordered_ind
        ordered_features = df_mean_zscore.columns[row_order]

        # Customize plot
        g.ax_heatmap.set_yticks(np.arange(len(row_order)) + 0.5)
        g.ax_heatmap.set_yticklabels(ordered_features, rotation=0, fontsize=10)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right")
        plt.title("Group-Level Mean Z-scores with Significance Markers")
        plt.savefig(dest_path, dpi=300, bbox_inches="tight")
        plt.close(g.fig)

    # Step 3: If grouping_mode is "individual", create an individual-level heatmap
    elif grouping_mode == "individual":
        # Generate group-level order of features for consistency
        df_mean_zscore = df_zscore.groupby(group_variable).mean()
        g = sns.clustermap(
            df_mean_zscore.T,
            cmap="inferno",
            center=0,
            metric="euclidean",
            method="ward",
        )
        row_order = g.dendrogram_row.reordered_ind
        ordered_features = df_mean_zscore.columns[row_order]

        # Prepare the individual-level dataframe and color mapping
        df_individual = df_zscore[ordered_features].copy()
        df_individual[group_variable] = df_zscore[group_variable]
        lut = dict(zip(df[group_variable].unique(), sns.color_palette("husl", len(df[group_variable].unique()))))
        group_colors = df_individual[group_variable].map(lut)
        df_individual.drop(columns=[group_variable], inplace=True)

        # Create the individual-level heatmap with clustering
        g_ind = sns.clustermap(
            df_individual.T,
            cmap="inferno",
            center=0,
            col_colors=group_colors,
            metric="euclidean",
            method="ward",
        )
        g_ind.ax_heatmap.set_yticks(np.arange(len(ordered_features)) + 0.5)
        g_ind.ax_heatmap.set_yticklabels(ordered_features, rotation=0, fontsize=10)
        g_ind.ax_heatmap.set_xticklabels(g_ind.ax_heatmap.get_xmajorticklabels(), rotation=45, fontsize=10, ha="right")

        # Add legend for group colors
        for label in df[group_variable].unique():
            g_ind.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
        g_ind.ax_col_dendrogram.legend(title=group_variable, loc="center", ncol=len(df[group_variable].unique()))

        plt.title("Individual Z-scores (Sorted by Clustering)")
        plt.savefig(dest_path, dpi=300, bbox_inches="tight")
        plt.close(g_ind.fig)

    return
