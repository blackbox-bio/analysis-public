from typing import List, Tuple

import pandas as pd

from utils import *
from palmreader_analysis import SummaryContext


def generate_summary_generic(features_files: List[str], time_bin=(0, -1)):
    contexts: List[SummaryContext] = []

    for file in features_files:
        contexts.append(SummaryContext(file, time_bin))

    for context in contexts:
        for column in SummaryContext.get_all_columns():
            column.summarize(context)

        # TODO: remove this once the name map is done at the computation level
        context.finish()

    return SummaryContext.merge_to_df(contexts)


def _df_concat_step(prev, next):
    if prev is None:
        return next

    return pd.concat([prev, next])


def generate_summaries_generic(
    features_files: List[str], time_bins: List[Tuple[float, float]]
):
    df = None

    for time_bin in time_bins:
        df = _df_concat_step(df, generate_summary_generic(features_files, time_bin))

    return df


RATIO_DECIMALS = 4
DEFAULT_DECIMALS = 2


def _is_ratio_column(column: str) -> bool:
    # relative_ covers the legacy relative columns, which have no "(ratio)" suffix.
    # don't match on "ratio" alone: "bin duration (min)" contains it
    return "relative_" in column or "_ratio" in column or "(ratio" in column


def write_summary_csv(df: pd.DataFrame, summary_path):
    """
    Write a summary dataframe to csv, with ratio and relative columns at
    RATIO_DECIMALS and all other float columns at DEFAULT_DECIMALS.
    """
    df = df.copy()

    for column in filter(_is_ratio_column, df.columns):
        df[column] = df[column].map(
            lambda value: f"{value:.{RATIO_DECIMALS}f}" if pd.notna(value) else value
        )

    df.to_csv(summary_path, float_format=f"%.{DEFAULT_DECIMALS}f")


def generate_summary_csv(analysis_folder, time_bins):
    """
    Generate summary csv from the processed recordings
    """
    recording_list = get_recording_list([analysis_folder])
    summary_dest = os.path.join(analysis_folder, "summary.csv")

    features_files = [
        os.path.join(recording, "features.h5") for recording in recording_list
    ]

    df = generate_summaries_generic(features_files, time_bins)

    write_summary_csv(df, summary_dest)
