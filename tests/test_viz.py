"""Smoke tests for draw_matches visualisation."""

import numpy as np

from matchability import distortions as D
from matchability.matchers.classical import ClassicalMatcher
from matchability.metric import compute_matchability
from matchability.viz import draw_matches


def test_draw_matches_returns_side_by_side_image(stereo_pair):
    left, right_gt = stereo_pair
    right_pred = D.apply("gaussian_blur", right_gt, 4.0)
    result = compute_matchability(left, right_gt, right_pred, matcher=ClassicalMatcher())
    out = draw_matches(left, right_pred, result, title="test")
    h, w = left.shape[:2]
    assert out.shape[0] == h
    assert out.shape[1] >= 2 * w
    assert out.dtype == np.uint8
