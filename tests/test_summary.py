import os
import shutil
from .helpers import get_sample_data, compare_csv_files, get_testdata_root
from api import summary_v1, summary_v2, summary_v3


def summary_file(filename):
    return get_testdata_root() / "summaries" / filename


SINGLE_BINLESS = summary_file("single_binless.csv")
SINGLE_BINNED = summary_file("single_binned.csv")


def test_summary_v1(tmp_path):
    sample = get_sample_data()[0]

    # v1 expects a directory with all features files. it unconditionally
    # includes and .h5 files so we can't give it the sample data root because
    # that has the DeepLabCut h5 files which will raise errors
    src_dir = tmp_path / "features"
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy(sample.features_h5, src_dir)

    dest_path = tmp_path / "summary.csv"

    summary_v1(
        {
            "features_dir": src_dir,
            "summary_path": dest_path,
        }
    )

    compare_csv_files(
        SINGLE_BINLESS,
        dest_path,
    )


def test_summary_v2(tmp_path):
    sample = get_sample_data()[0]

    dest_path = tmp_path / "summary.csv"

    summary_v2(
        {
            "features_files": [sample.features_h5],
            "summary_path": dest_path,
        }
    )

    compare_csv_files(
        SINGLE_BINLESS,
        dest_path,
    )


def test_summary_v3_single_binless(tmp_path):
    sample = get_sample_data()[0]

    dest_path = tmp_path / "summary.csv"

    summary_v3(
        {
            "features_files": [sample.features_h5],
            "summary_path": dest_path,
            "time_bins": [(0, -1)],
        }
    )

    compare_csv_files(
        SINGLE_BINLESS,
        dest_path,
    )


def test_summary_v3_single_binned(tmp_path):
    sample = get_sample_data()[0]

    dest_path = tmp_path / "summary.csv"

    summary_v3(
        {
            "features_files": [sample.features_h5],
            "summary_path": dest_path,
            "time_bins": [
                (0, 2),
                (2, 4),
                (4, -1),
            ],
        }
    )

    compare_csv_files(
        SINGLE_BINNED,
        dest_path,
    )
