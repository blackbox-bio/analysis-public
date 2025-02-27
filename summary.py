from typing import List, Tuple

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

    df.to_csv(summary_dest, float_format="%.2f")
