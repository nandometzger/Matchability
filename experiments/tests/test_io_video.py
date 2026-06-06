"""Integration test for MV-HEVC extraction (skips when no sample video is present)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.lib.io_video import extract_stereo_frame

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "raw" / "0001.mov"

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists() or shutil.which("ffmpeg") is None,
    reason="sample MV-HEVC video or ffmpeg not available",
)


def test_extract_stereo_frame_returns_rectified_pair():
    left, right = extract_stereo_frame(SAMPLE, frame_index=0)
    assert left.shape == right.shape
    assert left.ndim == 3 and left.shape[2] == 3
    assert np.abs(left.astype(int) - right.astype(int)).mean() > 1.0
