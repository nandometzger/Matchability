"""Tests for the ``matchability`` console entry point."""

import pytest

from matchability.cli import main


def test_cli_reports_metric_for_classical_backend(tmp_path, capsys, stereo_pair):
    pil = pytest.importorskip("PIL.Image")
    left, right_gt = stereo_pair
    lp, rp = tmp_path / "l.png", tmp_path / "r.png"
    pil.fromarray(left).save(lp)
    pil.fromarray(right_gt).save(rp)

    rc = main([str(lp), str(rp), str(rp), "--backend", "classical"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "E_match" in out
    assert "TP=" in out


def test_cli_writes_visualization(tmp_path, stereo_pair):
    pil = pytest.importorskip("PIL.Image")
    left, right_gt = stereo_pair
    lp, rp = tmp_path / "l.png", tmp_path / "r.png"
    pil.fromarray(left).save(lp)
    pil.fromarray(right_gt).save(rp)
    viz = tmp_path / "viz.png"

    rc = main([str(lp), str(rp), str(rp), "--backend", "classical", "--viz", str(viz)])
    assert rc == 0
    assert viz.exists() and viz.stat().st_size > 0
