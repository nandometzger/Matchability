"""Smoke tests for visualisation helpers."""

import numpy as np

from matchability import distortions as D
from matchability.matchers.classical import ClassicalMatcher
from matchability.metric import compute_matchability
from matchability.viz import draw_matches, plot_distortion_grid


def test_draw_matches_returns_side_by_side_image(stereo_pair):
    left, right_gt = stereo_pair
    right_pred = D.apply("gaussian_blur", right_gt, 4.0)
    result = compute_matchability(left, right_gt, right_pred, matcher=ClassicalMatcher())
    out = draw_matches(left, right_pred, result, title="test")
    h, w = left.shape[:2]
    assert out.shape[0] == h
    assert out.shape[1] >= 2 * w  # two panels side by side
    assert out.dtype == np.uint8


def test_plot_distortion_grid_writes_file(tmp_path):
    data = {
        "gaussian_blur": {
            "severities": [0.5, 2.0, 8.0],
            "error": [0.1, 0.5, 0.9],
            "ssim": [0.95, 0.7, 0.4],
            "trend": "rises",
        },
        "horizontal_shift": {
            "severities": [1, 8, 32],
            "error": [0.0, 0.02, 0.05],
            "ssim": [0.9, 0.6, 0.3],
            "trend": "flat",
        },
    }
    out = plot_distortion_grid(data, tmp_path / "grid.png", title="sweep")
    assert out.exists() and out.stat().st_size > 0
