"""Unit tests for the matcher-agnostic metric pipeline (uses a deterministic mock)."""

import numpy as np
import pytest

from matchability.matchers.mock import MockMatcher
from matchability.metric import compute_matchability
from matchability.types import LeftToRightMatches


def _img(tag: int) -> np.ndarray:
    """A tiny canonical image whose first pixel encodes a key for the mock."""
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[0, 0, 0] = tag
    return a


def test_pipeline_applies_epipolar_confidence_and_validity():
    # 5 fixed reference keypoints in the left image (x=100, rows 10..50).
    kpts = np.array([[100, 10], [100, 20], [100, 30], [100, 40], [100, 50]], dtype=float)

    gt = LeftToRightMatches(
        right_xy=np.array([[90, 10], [90, 20], [90, 30], [90, 45], [0, 0]], dtype=float),
        confidence=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        valid=np.array([1, 1, 1, 1, 0], dtype=bool),
    )
    # kp3 has dy=5 (epipolar-inconsistent), kp4 invalid -> M_gt = {0,1,2}
    pred = LeftToRightMatches(
        right_xy=np.array([[90, 10], [90, 20], [90, 31], [90, 40], [90, 50]], dtype=float),
        confidence=np.array([1.0, 0.1, 1.0, 1.0, 1.0]),
        valid=np.array([1, 1, 1, 1, 1], dtype=bool),
    )
    # kp1 below conf threshold; kp3,kp4 consistent in pred only -> M_pred = {0,2,3,4}
    matcher = MockMatcher(keypoints=kpts, responses={1: gt, 2: pred})

    r = compute_matchability(
        _img(0), _img(1), _img(2), matcher=matcher, tau=2.0, conf_threshold=0.5
    )

    assert (r.tp, r.fn, r.fp) == (2, 1, 2)  # TP={0,2}, FN={1}, FP={3,4}
    assert r.error == pytest.approx((2 + 1) / (2 + 2 + 1))  # (FP+FN)/union = 0.6
    assert r.n_reference == 5


def test_identical_predictions_give_zero_error():
    kpts = np.array([[100, 10], [100, 20]], dtype=float)
    same = LeftToRightMatches(
        right_xy=np.array([[90, 10], [90, 20]], dtype=float),
        confidence=np.ones(2),
        valid=np.ones(2, dtype=bool),
    )
    matcher = MockMatcher(keypoints=kpts, responses={1: same, 2: same})
    r = compute_matchability(_img(0), _img(1), _img(2), matcher=matcher, tau=2.0)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.error == 0.0


def test_no_reference_keypoints_is_zero_error():
    empty = LeftToRightMatches(
        right_xy=np.zeros((0, 2)), confidence=np.zeros(0), valid=np.zeros(0, dtype=bool)
    )
    matcher = MockMatcher(keypoints=np.zeros((0, 2)), responses={1: empty, 2: empty})
    r = compute_matchability(_img(0), _img(1), _img(2), matcher=matcher, tau=2.0)
    assert r.n_reference == 0
    assert r.error == 0.0
