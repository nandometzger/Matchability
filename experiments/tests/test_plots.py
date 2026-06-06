"""Smoke tests for sensitivity plot functions."""

from experiments.lib.plots import _degradation, plot_distortion_grid, plot_metric_comparison_grid


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
    assert _degradation([0.0, 5.0, 10.0], 0.0, 10.0, higher_is_worse=True) == [0.0, 0.5, 1.0]
    assert _degradation([1.0, 0.5, 0.0], 0.0, 1.0, higher_is_worse=False) == [0.0, 0.5, 1.0]
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
