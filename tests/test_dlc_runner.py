import pytest

pytest.importorskip("deeplabcut")

import cv2
import torch

from dlc_runner import limit_cpu_threads


def test_limit_cpu_threads_caps_and_restores():
    torch_threads, cv2_threads = torch.get_num_threads(), cv2.getNumThreads()

    with limit_cpu_threads(1):
        assert torch.get_num_threads() == 1
        assert cv2.getNumThreads() == 1

    assert torch.get_num_threads() == torch_threads
    assert cv2.getNumThreads() == cv2_threads


def test_limit_cpu_threads_restores_after_exception():
    torch_threads, cv2_threads = torch.get_num_threads(), cv2.getNumThreads()

    with pytest.raises(RuntimeError):
        with limit_cpu_threads(1):
            raise RuntimeError("inference failed")

    assert torch.get_num_threads() == torch_threads
    assert cv2.getNumThreads() == cv2_threads
