import os
from summary_viz import *


# user input: load the summary.csv file
experiment_folder = "/Users/zihealexzhang/work_local/blackbox_data/test/test_viz/"
test_df = os.path.join(experiment_folder, "test_df.csv")
test_df = pd.read_csv(test_df)

# --------------------------------
# user input for selecting columns to be included in the plot

# (a list of columns) -> user input for selecting columns to be included in the plot, default would be all numerical
# (features) columns except for the recording_time
selected_columns = [
    "distance_traveled (pixel)",
    "standing_on_two_hind_paws (ratio of time)",
    "average_hind_left_luminance",
    "average_hind_right_luminance",
    "average_front_left_luminance",
    "average_front_right_luminance",
]

# (a boolean vector mask) -> user input for selecting rows (a subset of the recordings) to be included in the plot,
# default would be all recordings selected_rows = [True, True, True, True, True, True, True, True, True, True] user
# can select a subset of the recordings based on the values in the grouping variables (e.g., pain_model, treatment)
selected_rows = test_df["Treatment"] == "Veh"
# user can manually select each row to be included in the plot via the user interface
# selected_rows = [True, False, True, False, True, False, True, False, True, False]

# (a string for the variable name) -> user input for selecting the grouping variable
group_variable = "pain_model"

# (a True/False boolean value) -> user input for whether ranking the columns by statistical significance between groups
sort_by_significance = True

# --------------------------------

# preprocess the summary csv file for visualization with user inputs
df_plot = summary_viz_preprocess(
    df=test_df,
    rows_to_include=selected_rows,
    columns_to_include=selected_columns,
    group_variable=group_variable,
)


# generate the PairGrid plot
generate_PairGrid_plot(
    df=df_plot, group_variable=group_variable, sort_by_significance=sort_by_significance
)

# generate the bar plots
generate_bar_plots(
    df=df_plot, group_variable=group_variable, sort_by_significance=sort_by_significance
)


# generate joint plot for a pair of columns
# user input for selecting a pair of columns to be included in the joint plot
g = sns.jointplot(
    data=df_plot,
    x="distance_traveled (pixel)",
    y="standing_on_two_hind_paws (ratio of time)",
    hue=group_variable,
)
