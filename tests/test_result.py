"""Unit tests for the MatchabilityResult decomposition (pure algebra, no matcher)."""

import numpy as np
import pytest

from matchability.types import MatchabilityResult


def test_from_masks_decomposition():
    # reference keypoints:    0  1  2  3  4
    m_gt = np.array([1, 1, 1, 0, 0], dtype=bool)
    m_pred = np.array([1, 0, 1, 1, 0], dtype=bool)
    r = MatchabilityResult.from_masks(m_gt, m_pred)
    assert r.tp == 2  # idx 0, 2 matchable in both
    assert r.fn == 1  # idx 1 matchable in GT only -> omission
    assert r.fp == 1  # idx 3 matchable in pred only -> hallucination
    assert r.n_reference == 5
    assert r.error == pytest.approx((1 + 1) / (2 + 1 + 1))  # 0.5
    assert r.error_pct == pytest.approx(50.0)
    assert r.jaccard == pytest.approx(2 / 4)


def test_empty_union_is_zero_error():
    z = np.zeros(5, dtype=bool)
    r = MatchabilityResult.from_masks(z, z)
    assert (r.tp, r.fp, r.fn) == (0, 0, 0)
    assert r.error == 0.0  # convention: IoU of two empty sets is 1 -> error 0
    assert r.jaccard == 1.0


def test_paper_figure10_example_reproduces_40_9():
    # Elastic3D Fig. 10 "Ours": TP=4717, FP=1908, FN=1354 -> E_match = 40.9
    r = MatchabilityResult(tp=4717, fp=1908, fn=1354, n_reference=4717 + 1908 + 1354)
    assert round(r.error_pct, 1) == 40.9


def test_from_masks_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MatchabilityResult.from_masks(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_error_is_fraction_pct_is_scaled():
    r = MatchabilityResult(tp=1, fp=1, fn=2, n_reference=4)
    assert 0.0 <= r.error <= 1.0
    assert r.error_pct == pytest.approx(100.0 * r.error)
