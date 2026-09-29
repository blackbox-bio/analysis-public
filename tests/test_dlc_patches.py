import pytest
import torch

pytest.importorskip("deeplabcut")

from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
    HeatmapPredictor,
)

import dlc_patches

# the original, before any patch is applied in this process
ORIGINAL_GET_POSE_PREDICTION = HeatmapPredictor.get_pose_prediction


@pytest.mark.parametrize("batch_size", [1, 8])
# 65 is the heatmap size for 512x512 frames, 87 for 1380x1380
@pytest.mark.parametrize("size", [65, 87])
@pytest.mark.parametrize("with_locref", [True, False])
def test_get_pose_prediction_matches_original(batch_size, size, with_locref):
    torch.manual_seed(0)
    num_joints = 57
    predictor = HeatmapPredictor(location_refinement=with_locref, locref_std=7.2801)

    heatmap = torch.rand(batch_size, size, size, num_joints)
    locref = torch.randn(batch_size, size, size, num_joints, 2) if with_locref else None
    scale_factors = (16, 16)

    expected = ORIGINAL_GET_POSE_PREDICTION(predictor, heatmap, locref, scale_factors)
    actual = dlc_patches.get_pose_prediction(predictor, heatmap, locref, scale_factors)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_patch_skipped_for_unknown_version(monkeypatch):
    monkeypatch.setattr(HeatmapPredictor, "get_pose_prediction", ORIGINAL_GET_POSE_PREDICTION)
    monkeypatch.setattr(dlc_patches, "_GET_POSE_PREDICTION_HASHES", {})

    assert dlc_patches.apply_dlc_patches() is False
    assert HeatmapPredictor.get_pose_prediction is ORIGINAL_GET_POSE_PREDICTION


def test_patch_applied_for_known_version(monkeypatch):
    monkeypatch.setattr(HeatmapPredictor, "get_pose_prediction", ORIGINAL_GET_POSE_PREDICTION)

    assert dlc_patches.apply_dlc_patches() is True
    assert HeatmapPredictor.get_pose_prediction is dlc_patches.get_pose_prediction
    # applying twice is a no-op
    assert dlc_patches.apply_dlc_patches() is True
