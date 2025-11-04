import pandas as pd
from palmreader_analysis.events import Palmreader
import numpy as np

def _listify(iterable):
    return "- " + '\n- '.join(map(str, iterable))


def summary_qc(
        df,
        group_variable: str,
):
    """
    for any summary statistical modeling,
    this function take the summary daraframe and the grouping variable,
    then run a quality control process to flag and remove all problematic feature columns,
    then return the dataframe
    """

    # check for feature columns that have non-numerical values
    non_numerical_columns = df.select_dtypes(exclude="number").columns
    # drop non-numerical columns except the group variable
    if group_variable in non_numerical_columns:
        non_numerical_columns = non_numerical_columns.drop(group_variable)

    if len(non_numerical_columns) > 0:
        Palmreader.warning("Some readouts have non-numerical values and have been omitted.",
                           f"The following columns have been ommitted:\n{_listify(non_numerical_columns)}")
        print(f"Warning: The following summary readouts have non-numerical values: {non_numerical_columns}.")
        print("These columns will be excluded from the cluster heatmap plot.")

        df = df.drop(columns=non_numerical_columns)

    # check and drop 'total recording_time (min)' column if it exists
    if "total recording_time (min)" in df.columns:
        df = df.drop(columns="total recording_time (min)")

    # check for feature columns that have missing values
    missing_values = df.columns[df.isnull().any()]
    if len(missing_values) > 0:
        Palmreader.warning(f"Some readouts have missing values and have been omitted.",
                           f"The following columns have been ommitted:\n{_listify(missing_values)}")
        print(f"Warning: The following summary readouts have missing values: {missing_values}.")
        print("These columns will be excluded from the cluster heatmap plot.")

        df = df.drop(columns=missing_values)

    # check for feature columns that have inf values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_values = numeric_df.columns[np.isinf(numeric_df.to_numpy()).any(axis=0)]
    if len(inf_values) > 0:
        Palmreader.warning(f"Some readouts have infinity values and have been omitted.",
                           f"The following columns have been ommitted:\n{_listify(inf_values)}")
        print(f"Warning: The following summary readouts have infinity values: {inf_values}.")
        print("These columns will be excluded from the cluster heatmap plot.")

        df = df.drop(columns=inf_values)

    # check for feature columns that have constant values
    constant_values = df.columns[df.nunique() == 1]

    # check if the group variable is in the constant values, if so, remove it from the list
    if group_variable in constant_values:
        constant_values = constant_values.drop(group_variable)
        # TODO: deal with summary df with just one group in the group variable


    if len(constant_values) > 0:
        Palmreader.warning(f"Some readouts have constant values and have been omitted.",
                           f"The following columns have been ommitted:\n{_listify(constant_values)}")
        print(f"Warning: The following summary readouts have constant values: {constant_values}.")
        print("These columns will be excluded from the cluster heatmap plot.")

        df = df.drop(columns=constant_values)



    return df