"""Unit tests for sensitivity aggregation + summary rendering."""

import pytest

from experiments.lib.report import aggregate, write_summary


def _row(distortion, severity, error, ssim, psnr, trend="rises"):
    return {
        "distortion": distortion,
        "trend": trend,
        "severity": severity,
        "error": error,
        "ssim": ssim,
        "psnr": psnr,
    }


def test_aggregate_means_over_videos_and_keeps_sweep_order():
    rows = [
        _row("blur", 1.0, 0.2, 0.9, 30),
        _row("blur", 1.0, 0.4, 0.8, 20),
        _row("blur", 2.0, 0.8, 0.5, 15),
    ]
    per = aggregate(rows)
    assert per["blur"]["severities"] == [1.0, 2.0]  # first-seen order
    assert per["blur"]["error"][0] == pytest.approx(0.3)  # mean(0.2, 0.4)
    assert per["blur"]["psnr"][0] == pytest.approx(25.0)
    assert per["blur"]["trend"] == "rises"


def test_aggregate_caps_infinite_psnr():
    per = aggregate([_row("identity", 0.0, 0.0, 1.0, float("inf"), trend="anchor_low")])
    assert per["identity"]["psnr"][0] == 100.0


def test_write_summary_includes_psnr(tmp_path):
    per = {
        "blur": {
            "severities": [1, 2],
            "error": [0.1, 0.9],
            "ssim": [0.9, 0.4],
            "psnr": [30, 15],
            "trend": "rises",
        }
    }
    path = tmp_path / "summary.md"
    write_summary(per, path, "meta line")
    text = path.read_text()
    assert "PSNR" in text and "meta line" in text and "blur" in text
