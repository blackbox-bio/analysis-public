import argparse
from enum import Enum
from typing import TypedDict, List
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

    run_deeplabcut(config_path, videos)

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
    from summary import generate_summary_v1

    features_dir = args['features_dir']
    summary_path = args['summary_path']

    generate_summary_v1(features_dir, summary_path)

class SummaryArgsV2(TypedDict):
    features_files: List[str]
    summary_path: str

def summary_v2(args: SummaryArgsV2):
    from summary import generate_summary_v2

    features_files = args['features_files']
    summary_path = args['summary_path']

    generate_summary_v2(features_files, summary_path)

# Palmreader <-> Analysis API
# The following code is relied upon by the Palmreader software. Take special care when modifying it.
class ApiFunction(Enum):
    DEEPLABCUT = 'deeplabcut'
    FEATURES = 'features'
    SUMMARY = 'summary'

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
        else:
            summary_v2(api_args)

if __name__ == '__main__':
    main()
