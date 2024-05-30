# globally import things that all functions share
# don't import anything not strictly required for the API itself. import API specific things in the functions themselves
import argparse
from enum import Enum
from typing import TypedDict, List, Tuple
import json

# API functions
class DeepLabCutArgs(TypedDict):
    config_path: str
    videos: List[str]

def deeplabcut(args: DeepLabCutArgs):
    # only import code that depends on deeplabcut if we're actually going to use it
    from dlc_runner import run_deeplabcut

    config_path = args['config_path']
    videos = args['videos']

    run_deeplabcut(config_path, videos, False)

class Extraction(TypedDict):
    name: str
    ftir_path: str
    tracking_path: str
    dest_path: str

class Extractions(TypedDict):
    extractions: List[Extraction]

def features(args: Extractions):
    from process import extract_features

    extractions = args['extractions']

    for extraction in extractions:
        extract_features(
            extraction['name'],
            extraction['ftir_path'],
            extraction['tracking_path'],
            extraction['dest_path']
        )

class SummaryArgsV1(TypedDict):
    features_dir: str
    summary_path: str

def summary_v1(args: SummaryArgsV1):
    """
    v1 API expects all features to be in a single folder. this function collects all .h5 files in the given folder and uses them
    """
    import os
    from summary import generate_summary_generic

    features_dir = args['features_dir']
    summary_path = args['summary_path']

    features_files = []

    for file in os.listdir(features_dir):
        if file.endswith(".h5"):
            features_files.append(os.path.join(features_dir, file))

    df = generate_summary_generic(features_files)

    df.to_csv(summary_path, float_format="%.2f")

class SummaryArgsV2(TypedDict):
    features_files: List[str]
    summary_path: str

def summary_v2(args: SummaryArgsV2):
    """
    v2 API expects a list of .h5 files. this function uses them directly
    """
    from summary import generate_summary_generic

    features_files = args['features_files']
    summary_path = args['summary_path']

    df = generate_summary_generic(features_files)

    df.to_csv(summary_path, float_format="%.2f")

class SummaryArgsV3(TypedDict):
    features_files: List[str]
    summary_path: str
    time_bins: List[Tuple[float, float]]

def summary_v3(args: SummaryArgsV3):
    """
    v3 API is v2 with time bins
    """
    from summary import generate_summaries_generic

    features_files = args['features_files']
    summary_path = args['summary_path']
    time_bins = args['time_bins']

    df = generate_summaries_generic(features_files, time_bins)

    df.to_csv(summary_path, float_format="%.2f")

class SkeletonArgs(TypedDict):
    config_path: str
    videos: List[str]

def skeleton(args: SkeletonArgs):
    from dlc_runner import generate_skeleton

    config_path = args['config_path']
    videos = args['videos']

    generate_skeleton(config_path, videos)

class PairGridArgs(TypedDict):
    # graph arguments arrive in camelCase because of a Palmreader optimization
    summaryPath: str
    enabledRows: List[bool]
    vars: List[str]
    hue: str
    diagKind: str
    upperKind: str
    lowerKind: str
    destPath: str

def pair_grid(args: PairGridArgs):
    import pandas as pd
    import numpy as np
    from summary_viz import summary_viz_preprocess, generate_PairGrid_plot

    summary_path = args['summaryPath']
    enabled_rows = args['enabledRows']
    vars = args['vars']
    hue = args['hue']
    diag_kind = args['diagKind']
    upper_kind = args['upperKind']
    lower_kind = args['lowerKind']
    dest_path = args['destPath']

    df = pd.read_csv(summary_path)

    df = summary_viz_preprocess(df, enabled_rows, vars, hue)

    generate_PairGrid_plot(
        df,
        hue,
        diag_kind,
        upper_kind,
        lower_kind,
        dest_path,
    )

class BarPlotsArgs(TypedDict):
    # graph arguments arrive in camelCase because of a Palmreader optimization
    summaryPath: str
    enabledRows: List[bool]
    vars: List[str]
    hue: str
    sortBySignificance: bool
    destPath: str

def bar_plots(args: BarPlotsArgs):
    import pandas as pd
    import numpy as np
    from summary_viz import summary_viz_preprocess, generate_bar_plots

    summary_path = args['summaryPath']
    enabled_rows = args['enabledRows']
    vars = args['vars']
    hue = args['hue']
    sort_by_significance = args['sortBySignificance']
    dest_path = args['destPath']

    df = pd.read_csv(summary_path)

    df = summary_viz_preprocess(df, enabled_rows, vars, hue)

    generate_bar_plots(
        df,
        hue,
        dest_path,
        sort_by_significance
    )


# Palmreader <-> Analysis API
# The following code is relied upon by the Palmreader software. Take special care when modifying it.
class ApiFunction(Enum):
    DEEPLABCUT = 'deeplabcut'
    FEATURES = 'features'
    SUMMARY = 'summary'
    SKELETON = 'skeleton'
    PAIRGRID = 'pairgrid'
    BAR_PLOTS = 'bar_plots'

    def __str__(self):
        return self.value

def main():
    p = argparse.ArgumentParser()

    # the function to call, one of the ApiFunction variants
    p.add_argument('--function', type=ApiFunction, choices=list(ApiFunction), required=True, dest='function')
    # a JSON literal that will be parsed into function arguments for the given function
    # the structure of this literal depends on the function being called
    p.add_argument('--args', type=str, required=True, dest='args')
    # the version of the API to use. defaults to 1 for backwards compatibility
    p.add_argument('--api-version', type=int, default=1, dest='api_version')

    args = p.parse_args()

    api_args = json.loads(args.args)
    if args.function == ApiFunction.DEEPLABCUT:
        deeplabcut(api_args)
    elif args.function == ApiFunction.FEATURES:
        features(api_args)
    elif args.function == ApiFunction.SUMMARY:
        if args.api_version == 1:
            summary_v1(api_args)
        elif args.api_version == 2:
            summary_v2(api_args)
        else:
            summary_v3(api_args)
    elif args.function == ApiFunction.SKELETON:
        skeleton(api_args)
    elif args.function == ApiFunction.PAIRGRID:
        pair_grid(api_args)
    elif args.function == ApiFunction.BAR_PLOTS:
        bar_plots(api_args)

if __name__ == '__main__':
    main()
