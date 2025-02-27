from .helpers import get_sample_data, compare_h5_files


def test_feature_extraction(tmp_path):
    from api import features

    dest_path = tmp_path / "features.h5"
    # TODO: iterate over multiple samples
    sample = get_sample_data()[0]

    features(
        {
            "extractions": [
                {
                    "name": "sample1",
                    "ftir_path": sample.ftir_video,
                    "tracking_path": sample.tracking_filtered_h5,
                    "dest_path": dest_path,
                }
            ]
        }
    )

    compare_h5_files(sample.features_h5, dest_path)
