"""
Runtime performance patches for DeepLabCut inference.

Each patch replaces a DLC function with a faster version that gives identical
results. A patch is only applied if the DLC function's source matches a version
it was checked against (by sha256 of its source), so an unknown DLC version runs
unpatched instead of having code replaced that the patch was not written for.
"""

import hashlib
import inspect

import torch

# sha256 of the HeatmapPredictor.get_pose_prediction source for each DLC
# version this patch was checked against
_GET_POSE_PREDICTION_HASHES = {
    "281216273a6e81c2b64710e1469a670257a997a6b6cea35482387509ae4acf42": "3.0.0rc10",
}


def _source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def get_pose_prediction(self, heatmap, locref, scale_factors):
    """
    Drop-in replacement for HeatmapPredictor.get_pose_prediction.

    The original reads each frame's peak score and location refinement one
    body part at a time in a python loop (batch_size x num_joints tiny GPU
    operations per batch). This gathers them for the whole batch at once, with
    identical results.
    """
    y, x = self.get_top_values(heatmap)

    batch_size, height, width, num_joints = heatmap.shape

    # flat index of each joint's heatmap peak, shape (batch_size, num_joints)
    idx = y * width + x

    dz = torch.zeros((batch_size, 1, num_joints, 3), device=heatmap.device)
    dz[:, 0, :, 2] = heatmap.reshape(batch_size, height * width, num_joints).gather(
        1, idx.unsqueeze(1)
    )[:, 0]
    if locref is not None:
        locref = locref.reshape(batch_size, height * width, num_joints, 2)
        dz[:, 0, :, :2] = locref.gather(1, idx[:, None, :, None].expand(-1, 1, -1, 2))[
            :, 0
        ]

    # the rest is unchanged from the original
    x, y = torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)

    x = x * scale_factors[1] + 0.5 * scale_factors[1] + dz[:, :, :, 0]
    y = y * scale_factors[0] + 0.5 * scale_factors[0] + dz[:, :, :, 1]

    pose = torch.zeros((batch_size, 1, num_joints, 3), device=heatmap.device)
    pose[:, :, :, 0] = x
    pose[:, :, :, 1] = y
    pose[:, :, :, 2] = dz[:, :, :, 2]
    return pose


def apply_dlc_patches() -> bool:
    """
    Apply the patches to the installed DeepLabCut. Safe to call more than once.

    Returns True if the patch is active, False if it was skipped.
    """
    try:
        from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
            HeatmapPredictor,
        )
    except ImportError:
        print("DLC patch skipped: HeatmapPredictor not found in this DeepLabCut version")
        return False

    current = HeatmapPredictor.get_pose_prediction
    if current is get_pose_prediction:
        return True

    source_hash = _source_hash(current)
    if source_hash not in _GET_POSE_PREDICTION_HASHES:
        print(
            "DLC patch skipped: HeatmapPredictor.get_pose_prediction does not match a "
            f"checked version (sha256 {source_hash})"
        )
        return False

    HeatmapPredictor.get_pose_prediction = get_pose_prediction
    print(
        "DLC patch applied: HeatmapPredictor.get_pose_prediction "
        f"(checked against DLC {_GET_POSE_PREDICTION_HASHES[source_hash]})"
    )
    return True
