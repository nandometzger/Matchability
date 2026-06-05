"""The matcher-agnostic Matchability metric pipeline.

``compute_matchability`` is the core: detect one fixed reference keypoint set in
the left image, match it into the GT and predicted right views, keep only
confident epipolar-consistent matches, and decompose the two membership masks
into the Matchability Error (see ``docs/metric.md``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from matchability.epipolar import epipolar_consistent_rectified
from matchability.matchers.base import Matcher
from matchability.types import LeftToRightMatches, MatchabilityResult


def _membership_mask(
    keypoints_left: NDArray[np.floating],
    matches: LeftToRightMatches,
    tau: float,
    conf_threshold: float,
    disparity_bounds: tuple[float, float] | None,
) -> NDArray[np.bool_]:
    """Which reference keypoints have a confident, epipolar-consistent match."""
    confident = matches.valid & (matches.confidence >= conf_threshold)
    consistent = epipolar_consistent_rectified(
        keypoints_left, matches.right_xy, tau, disparity_bounds=disparity_bounds
    )
    return confident & consistent


def compute_matchability(
    left: NDArray[np.uint8],
    right_gt: NDArray[np.uint8],
    right_pred: NDArray[np.uint8],
    *,
    matcher: Matcher,
    tau: float = 2.0,
    conf_threshold: float = 0.0,
    disparity_bounds: tuple[float, float] | None = None,
    n_keypoints: int | None = None,
) -> MatchabilityResult:
    """Compute the Matchability Error for one ``(left, right_gt, right_pred)`` triplet.

    Args:
        left: left image (canonical ``(H, W, 3)`` uint8 RGB).
        right_gt: ground-truth right view.
        right_pred: predicted right view.
        matcher: the keypoint matcher backend (DeDoDe v2, classical, or mock).
        tau: epipolar vertical-disparity threshold (pixels, at the input resolution).
        conf_threshold: minimum match confidence to count a correspondence.
        disparity_bounds: optional ``(d_min, d_max)`` horizontal-disparity bound.
        n_keypoints: size of the fixed reference set (``None`` = matcher default).

    Returns:
        A :class:`~matchability.types.MatchabilityResult`.
    """
    keypoints_left = np.asarray(matcher.detect(left, n=n_keypoints), dtype=float).reshape(-1, 2)
    matches_gt = matcher.match(left, keypoints_left, right_gt)
    matches_pred = matcher.match(left, keypoints_left, right_pred)

    m_gt = _membership_mask(keypoints_left, matches_gt, tau, conf_threshold, disparity_bounds)
    m_pred = _membership_mask(keypoints_left, matches_pred, tau, conf_threshold, disparity_bounds)
    return MatchabilityResult.from_masks(
        m_gt,
        m_pred,
        keypoints_left=keypoints_left,
        right_xy_gt=matches_gt.right_xy,
        right_xy_pred=matches_pred.right_xy,
    )
