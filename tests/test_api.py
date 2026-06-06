"""Tests for the public, easy-to-use API (with an injected matcher -- no DeDoDe load)."""

import numpy as np
import pytest

import matchability
from matchability import Matchability, MatchabilityResult, matchability_error
from matchability import distortions as D
from matchability.matchers.classical import ClassicalMatcher


def test_public_exports_are_importable():
    assert hasattr(matchability, "matchability_error")
    assert hasattr(matchability, "Matchability")
    assert hasattr(matchability, "MatchabilityResult")


def test_class_identity_is_zero(stereo_pair):
    left, right_gt = stereo_pair
    metric = Matchability(matcher=ClassicalMatcher())
    res = metric(left, right_gt, right_gt)
    assert isinstance(res, MatchabilityResult)
    assert res.error == 0.0


def test_function_blur_is_high(stereo_pair):
    left, right_gt = stereo_pair
    right_pred = D.apply("gaussian_blur", right_gt, 8.0)
    res = matchability_error(left, right_gt, right_pred, matcher=ClassicalMatcher())
    assert res.error > 0.5


def test_accepts_mixed_input_types(tmp_path, stereo_pair):
    pil = pytest.importorskip("PIL.Image")
    left, right_gt = stereo_pair
    path = tmp_path / "pred.png"
    pil.fromarray(right_gt).save(path)
    metric = Matchability(matcher=ClassicalMatcher())
    # numpy array, PIL image, and a file path that all encode right_gt -> zero error
    res = metric(left, pil.fromarray(right_gt), str(path))
    assert res.error == 0.0


def test_mismatched_shapes_raise(stereo_pair):
    left, right_gt = stereo_pair
    bad = np.ascontiguousarray(right_gt[:, :200])  # different width
    with pytest.raises(ValueError):
        Matchability(matcher=ClassicalMatcher())(left, right_gt, bad)


def test_working_resolution_downscales(stereo_pair):
    left, right_gt = stereo_pair  # 256x256
    metric = Matchability(matcher=ClassicalMatcher(), working_resolution=64)
    # identical inputs -> zero error regardless, but exercises the resize path
    assert metric(left, right_gt, right_gt).error == 0.0
