"""Smoke tests for visualisation helpers."""

import numpy as np

from matchability import distortions as D
from matchability.matchers.classical import ClassicalMatcher
from matchability.metric import compute_matchability
from matchability.viz import (
    _degradation,
    draw_matches,
    plot_distortion_grid,
    plot_metric_comparison_grid,
)


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


def test_degradation_alignment_directions():
    # error rises with degradation -> stays as-is and rises 0..1
    assert _degradation([0.0, 5.0, 10.0], 0.0, 10.0, higher_is_worse=True) == [0.0, 0.5, 1.0]
    # ssim falls with degradation -> flipped so it also rises 0..1
    assert _degradation([1.0, 0.5, 0.0], 0.0, 1.0, higher_is_worse=False) == [0.0, 0.5, 1.0]
    # flat metric (no range) -> all zeros, no divide-by-zero
    assert _degradation([7.0, 7.0], 7.0, 7.0, higher_is_worse=True) == [0.0, 0.0]


def test_plot_metric_comparison_grid_writes_both_modes(tmp_path):
    data = {
        "gaussian_blur": {
            "severities": [1, 2, 3], "error": [0.1, 0.5, 0.9],
            "ssim": [0.9, 0.6, 0.3], "psnr": [30, 22, 15], "trend": "rises",
        },
        "horizontal_shift": {
            "severities": [1, 8, 32], "error": [0.0, 0.02, 0.05],
            "ssim": [0.9, 0.8, 0.7], "psnr": [35, 31, 28], "trend": "flat",
        },
    }
    for mode in ("sweep", "global"):
        out = plot_metric_comparison_grid(
            data, tmp_path / f"{mode}.png", normalize=mode, title=mode
        )
        assert out.exists() and out.stat().st_size > 0
