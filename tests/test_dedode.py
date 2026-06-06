"""Opt-in integration tests for the real DeDoDe v2 backend (downloads weights).

Run with ``pytest -m dedode``. These exercise the MPS/CPU path and verify the
faithful backend reproduces the paper's qualitative trends.
"""

import pytest

from matchability import distortions as D
from matchability.metric import compute_matchability

pytestmark = pytest.mark.dedode


@pytest.fixture(scope="module")
def dedode():
    from matchability.matchers.dedode import DeDoDeV2Matcher

    return DeDoDeV2Matcher()


def test_dedode_detect_runs(dedode, stereo_pair):
    left, _ = stereo_pair
    kp = dedode.detect(left, n=2000)
    assert kp.ndim == 2 and kp.shape[1] == 2 and len(kp) > 0


def _err(dedode, pair, name, severity):
    left, right_gt = pair
    right_pred = D.apply(name, right_gt, severity)
    return compute_matchability(
        left, right_gt, right_pred, matcher=dedode, tau=2.0, n_keypoints=2000
    ).error


def test_dedode_reproduces_paper_trends(dedode, stereo_pair):
    assert _err(dedode, stereo_pair, "identity", 0.0) < 0.05
    blur = _err(dedode, stereo_pair, "gaussian_blur", 8.0)
    shift = _err(dedode, stereo_pair, "horizontal_shift", 16.0)
    assert blur > 0.5  # texture loss penalised
    assert shift < 0.3  # translation tolerated
    assert blur > shift  # the headline decoupling
