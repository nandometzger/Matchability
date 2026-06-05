"""Property tests: does the metric behave the way the paper says it should?

These run the full pipeline with the real (classical) matcher on a synthetic
rectified stereo pair, and assert the qualitative trends from Elastic3D App. D.1
(plus extra controls). Thresholds carry wide margins to avoid flakiness.
"""

import pytest

from matchability import distortions as D
from matchability.matchers.classical import ClassicalMatcher
from matchability.metric import compute_matchability


@pytest.fixture(scope="module")
def matcher():
    return ClassicalMatcher()


def _err(matcher, pair, name, severity, tau=2.0):
    left, right_gt = pair
    right_pred = D.apply(name, right_gt, severity)
    return compute_matchability(left, right_gt, right_pred, matcher=matcher, tau=tau).error


def test_identity_prediction_is_zero_error(matcher, stereo_pair):
    assert _err(matcher, stereo_pair, "identity", 0.0) == 0.0


def test_horizontal_shift_stays_flat(matcher, stereo_pair):
    # DeDoDe/SIFT are translation-invariant; error must stay near zero across the sweep.
    errors = [
        _err(matcher, stereo_pair, "horizontal_shift", s)
        for s in D.get("horizontal_shift").severities
    ]
    assert max(errors) < 0.15


def test_vertical_shift_breaks_epipolar_consistency(matcher, stereo_pair):
    # A vertical shift beyond tau pushes matches off their epipolar line -> omissions.
    assert _err(matcher, stereo_pair, "vertical_shift", 6.0) > 0.5


def test_gaussian_blur_rises_sharply(matcher, stereo_pair):
    weak = _err(matcher, stereo_pair, "gaussian_blur", 0.5)
    strong = _err(matcher, stereo_pair, "gaussian_blur", 8.0)
    assert strong > 0.7
    assert strong >= weak  # monotone-ish: heavier blur is at least as bad


def test_blur_error_far_exceeds_equivalent_shift(matcher, stereo_pair):
    # The paper's headline: texture loss is penalised, geometric translation is not.
    blur = _err(matcher, stereo_pair, "gaussian_blur", 8.0)
    shift = _err(matcher, stereo_pair, "horizontal_shift", 32.0)
    assert blur > shift + 0.5


def test_occlusion_rises_with_area(matcher, stereo_pair):
    small = _err(matcher, stereo_pair, "occlusion_patch", 0.05)
    large = _err(matcher, stereo_pair, "occlusion_patch", 0.5)
    assert large > small
    assert large > 0.3


def test_brightness_gamma_more_robust_than_blur(matcher, stereo_pair):
    gamma = _err(matcher, stereo_pair, "brightness_gamma", 2.0)
    blur = _err(matcher, stereo_pair, "gaussian_blur", 8.0)
    assert gamma < blur  # photometric change is far more matchable than texture loss


def test_scramble_destroys_correspondence(matcher, stereo_pair):
    assert _err(matcher, stereo_pair, "scramble", 1.0) > 0.6
