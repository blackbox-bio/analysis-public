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
    df = df.dropna() # drop rows with missing values
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
