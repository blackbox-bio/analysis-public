# globally import things that all functions share
# don't import anything not strictly required for the API itself. import API specific things in the functions themselves
from palmreader import Palmreader, PalmreaderProgress
import argparse
from enum import Enum
from typing import TypedDict, List, Tuple, Literal
import json


# API functions
class DeepLabCutArgs(TypedDict):
    config_path: str
    videos: List[str]


def deeplabcut(args: DeepLabCutArgs):
    # only import code that depends on deeplabcut if we're actually going to use it
    from dlc_runner import run_deeplabcut

    config_path = args["config_path"]
    videos = args["videos"]

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

    extractions = args["extractions"]

    PalmreaderProgress.start_multi(len(extractions), "Extracting features")

    for extraction in extractions:
        # increment before each iteration so it's not zero indexed
        PalmreaderProgress.increment_multi()

        extract_features(
            extraction["name"],
            extraction["ftir_path"],
            extraction["tracking_path"],
            extraction["dest_path"],
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

    features_dir = args["features_dir"]
    summary_path = args["summary_path"]

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

    features_files = args["features_files"]
    summary_path = args["summary_path"]

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

    features_files = args["features_files"]
    summary_path = args["summary_path"]
    time_bins = args["time_bins"]

    df = generate_summaries_generic(features_files, time_bins)

    df.to_csv(summary_path, float_format="%.2f")


class SkeletonArgs(TypedDict):
    config_path: str
    videos: List[str]


def skeleton(args: SkeletonArgs):
    from dlc_runner import generate_skeleton

    config_path = args["config_path"]
    videos = args["videos"]

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

    summary_path = args["summaryPath"]
    enabled_rows = args["enabledRows"]
    vars = args["vars"]
    hue = args["hue"]
    diag_kind = args["diagKind"]
    upper_kind = args["upperKind"]
    lower_kind = args["lowerKind"]
    dest_path = args["destPath"]

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

    summary_path = args["summaryPath"]
    enabled_rows = args["enabledRows"]
    vars = args["vars"]
    hue = args["hue"]
    sort_by_significance = args["sortBySignificance"]
    dest_path = args["destPath"]

    df = pd.read_csv(summary_path)

    df = summary_viz_preprocess(df, enabled_rows, vars, hue)

    generate_bar_plots(df, hue, dest_path, sort_by_significance)


class ClusterHeatmapArgs(TypedDict):
    # graph arguments arrive in camelCase because of a Palmreader optimization
    summaryPath: str
    enabledRows: List[bool]
    vars: List[str]
    hue: str
    groupingMode: Literal["group", "individual"]
    destPath: str


def cluster_heatmap(args: ClusterHeatmapArgs):
    import pandas as pd
    import numpy as np
    from summary_viz import summary_viz_preprocess, generate_cluster_heatmap

    summary_path = args["summaryPath"]
    enabled_rows = args["enabledRows"]
    vars = args["vars"]
    hue = args["hue"]
    grouping_mode = args["groupingMode"]
    dest_path = args["destPath"]

    df = pd.read_csv(summary_path)

    df = summary_viz_preprocess(df, enabled_rows, vars, hue)

    generate_cluster_heatmap(df, hue, dest_path, grouping_mode)


# Palmreader <-> Analysis API
# The following code is relied upon by the Palmreader software. Take special care when modifying it.
class ApiFunction(Enum):
    DEEPLABCUT = "deeplabcut"
    FEATURES = "features"
    SUMMARY = "summary"
    SKELETON = "skeleton"
    PAIRGRID = "pairgrid"
    BAR_PLOTS = "bar_plots"
    CLUSTER_HEATMAP = "cluster_heatmap"

    def __str__(self):
        return self.value


def invoke_v2(func, args, task):
    Palmreader.set_enabled(True)

    try:
        func(args)
    except Exception as e:
        Palmreader.exception(f"An error occurred while {task}", e)


def main():
    p = argparse.ArgumentParser()

    # the function to call, one of the ApiFunction variants
    p.add_argument(
        "--function",
        type=ApiFunction,
        choices=list(ApiFunction),
        required=True,
        dest="function",
    )
    # a JSON literal that will be parsed into function arguments for the given function
    # the structure of this literal depends on the function being called
    p.add_argument("--args", type=str, required=True, dest="args")
    # the version of the API to use. defaults to 1 for backwards compatibility
    p.add_argument("--api-version", type=int, default=1, dest="api_version")
    # whether to enable API version 2. defaults to false for backwards compatibility
    p.add_argument("--v2", action="store_true", dest="v2")

    args = p.parse_args()

    api_args = json.loads(args.args)
    func = None
    task = ""

    if args.function == ApiFunction.DEEPLABCUT:
        func = deeplabcut
        task = "running DeepLabCut"
    elif args.function == ApiFunction.FEATURES:
        func = features
        task = "extracting features"
    elif args.function == ApiFunction.SUMMARY:
        task = "generating summary"

        if args.api_version == 1:
            func = summary_v1
        elif args.api_version == 2:
            func = summary_v2
        else:
            func = summary_v3
    elif args.function == ApiFunction.SKELETON:
        func = skeleton
        task = "generating skeleton video"
    elif args.function == ApiFunction.PAIRGRID:
        func = pair_grid
        task = "generating pair grid"
    elif args.function == ApiFunction.BAR_PLOTS:
        func = bar_plots
        task = "generating bar plots"
    elif args.function == ApiFunction.CLUSTER_HEATMAP:
        func = cluster_heatmap
        task = "generating cluster heatmap"

    if func is None:
        raise ValueError("Invalid function")

    if args.v2:
        invoke_v2(func, api_args, task)
    else:
        func(api_args)


if __name__ == "__main__":
    main()
