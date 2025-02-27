from typing import List
from dataclasses import dataclass
import os
import pathlib
import h5py
import numpy as np
import pandas as pd


@dataclass
class SampleData:
    @staticmethod
    def make_default(
        root: str,
        dlc_scorer: str = "DLC_resnet50_arcteryx500Nov4shuffle1_350000",
    ) -> "SampleData":
        spath = lambda x: os.path.join(root, x)
        dlcpath = lambda p, ext: os.path.join(root, f"{p}{dlc_scorer}{ext}")

        return SampleData(
            trans_video=spath("trans.avi"),
            ftir_video=spath("ftir.avi"),
            features_h5=spath("features.h5"),
            tracking_h5=dlcpath("trans", ".h5"),
            tracking_filtered_h5=dlcpath("trans", "_filtered.h5"),
            tracking_meta_pickle=dlcpath("trans", "_meta.pickle"),
        )

    # input data
    trans_video: str
    ftir_video: str

    # tracking data
    tracking_h5: str
    tracking_filtered_h5: str
    tracking_meta_pickle: str

    # output data
    features_h5: str


def get_testdata_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


def get_sample_data() -> List[SampleData]:
    data_root = get_testdata_root()

    sample1 = data_root / "sample1"

    return [
        SampleData.make_default(sample1.as_posix()),
    ]


def locate_array_differences(arr1: np.ndarray, arr2: np.ndarray):
    """
    Compare two numpy arrays that are known to be different and display a detailed message on how they differ.

    - If they differ slightly, the assertion message highlights mismatches.
    - If they are vastly different, the message provides a statistical summary.
    - Treats NaNs as equal.

    Parameters:
        arr1 (np.ndarray): First array.
        arr2 (np.ndarray): Second array.
    """
    if arr1.shape != arr2.shape:
        raise AssertionError(f"Shape mismatch: {arr1.shape} != {arr2.shape}")

    diff_mask = ~(np.equal(arr1, arr2) | (np.isnan(arr1) & np.isnan(arr2)))
    diff_indices = np.argwhere(diff_mask)

    num_diffs = len(diff_indices)
    total_elements = arr1.size
    diff_percentage = (num_diffs / total_elements) * 100

    if num_diffs == 0:
        return  # No differences

    if num_diffs > 300:
        raise AssertionError(
            f"Arrays differ in {diff_percentage:.2f}% of elements ({num_diffs}/{total_elements})."
        )

    diff_details = "\n".join(
        [
            f"Index {tuple(idx)}: {arr1[tuple(idx)]} != {arr2[tuple(idx)]}"
            for idx in diff_indices
        ]
    )

    raise AssertionError(f"Arrays differ at {num_diffs} elements:\n{diff_details}")


def compare_h5_files(reference_path: str, subject_path: str):
    reference = {}
    subject = {}

    def read_file(path: str, dest: dict):
        with h5py.File(path, "r") as hdf:
            assert len(hdf.keys()) == 1, "Expected only one group in the HDF5 file"

            group = list(hdf.keys())[0]
            for feature in hdf[group].keys():
                dest[feature] = np.array(hdf[group][feature])

    read_file(reference_path, reference)
    read_file(subject_path, subject)

    # check that the keys are the same in both directions
    reference_keys = set(reference.keys())
    subject_keys = set(subject.keys())

    if reference_keys != subject_keys:
        reference_missing = subject_keys - reference_keys
        subject_missing = reference_keys - subject_keys

        msg = "Key mismatch between files:\n"
        if reference_missing:
            msg += f"Keys in subject but not in reference: {reference_missing}\n"
        if subject_missing:
            msg += f"Keys in reference but not in subject: {subject_missing}\n"

        raise AssertionError(msg)

    for key in reference.keys():
        # we don't need to check if subject has this key because we already did

        if not np.allclose(reference[key], subject[key], equal_nan=True):
            try:
                # if they're not equal, we want the assertion to be more
                # helpful than "oh no, they're not equal". this function prints
                # out the differences, assuming that the arrays are not too
                # significantly different (100 elements or less)
                locate_array_differences(reference[key], subject[key])

                assert (
                    False
                ), "Arrays are not equal but no differences were found. This should not happen."
            except AssertionError as e:
                # the error raised by locate_array_differences does not know
                # which key we are currently comparing so we wrap the error in
                # another assertion error that does
                raise AssertionError(f"Mismatch in key {key}: {str(e)}")


def compare_csv_files(reference_path: str, subject_path: str):
    reference = pd.read_csv(reference_path, index_col=0)
    subject = pd.read_csv(subject_path, index_col=0)

    # check that the columns are the same in both directions
    reference_columns = set(reference.columns)
    subject_columns = set(subject.columns)

    if reference_columns != subject_columns:
        reference_missing = subject_columns - reference_columns
        subject_missing = reference_columns - subject_columns

        msg = "Column mismatch between files:\n"
        if reference_missing:
            msg += f"Columns in subject but not in reference: {reference_missing}\n"
        if subject_missing:
            msg += f"Columns in reference but not in subject: {subject_missing}\n"

        raise AssertionError(msg)

    # Ensure columns are in the same order
    reference = reference[sorted(reference_columns)]
    subject = subject[sorted(reference_columns)]

    # make sure that there are the same number of rows and each row has the
    # same index
    reference_index = set(reference.index)
    subject_index = set(subject.index)

    if reference_index != subject_index:
        reference_missing = subject_index - reference_index
        subject_missing = reference_index - subject_index

        msg = "Row index mismatch between files:\n"
        if reference_missing:
            msg += f"Row index in subject but not in reference: {reference_missing}\n"
        if subject_missing:
            msg += f"Row index in reference but not in subject: {subject_missing}\n"

        raise AssertionError(msg)

    # Align the rows to ensure correct order for comparison
    reference = reference.loc[sorted(reference_index)]
    subject = subject.loc[sorted(reference_index)]

    # Compare values
    differences = (reference != subject) & ~(reference.isna() & subject.isna())

    if differences.any().any():
        mismatches = np.where(differences)
        mismatch_details = []
        for row, col in zip(*mismatches):
            ref_value = reference.iloc[row, col]
            sub_value = subject.iloc[row, col]
            index_label = reference.index[row]
            column_label = reference.columns[col]
            mismatch_details.append(
                f"Row '{index_label}', Column '{column_label}': Reference={ref_value}, Subject={sub_value}"
            )

        raise AssertionError(
            "Data value mismatch found:\n" + "\n".join(mismatch_details)
        )
