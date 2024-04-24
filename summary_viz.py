from io import BytesIO
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import cv2 as cv
from enum import Enum

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

        sns.boxplot(
            x=group_variable,
            y=column,
            data=df,
            # fill = False,
            # split = True,
            legend = 'auto',
            # dodge=True,
        )

        plt.title(f"{column} grouped by {group_label}")
        plt.tight_layout()
        
        plot = plt.gcf()

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
        sort_by_significance=False
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
    
    g.savefig(dest_path, dpi=300)
