"""Epipolar-consistency tests for keypoint matches.

The metric keeps only matches that are consistent with the stereo epipolar
geometry. For *rectified* pairs (this package's primary case) the epipolar lines
are horizontal image rows, so consistency reduces to a small vertical-disparity
threshold ``|y_L - y_R| <= tau``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def epipolar_consistent_rectified(
    left_xy: NDArray[np.floating],
    right_xy: NDArray[np.floating],
    tau: float = 2.0,
    *,
    disparity_bounds: tuple[float, float] | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of epipolar-consistent matches for a rectified stereo pair.

    Args:
        left_xy: ``(K, 2)`` array of left keypoint ``(x, y)`` coordinates.
        right_xy: ``(K, 2)`` array of their matched right ``(x, y)`` coordinates.
        tau: maximum allowed absolute vertical disparity ``|y_L - y_R|`` (inclusive).
        disparity_bounds: optional ``(d_min, d_max)`` bound on the horizontal
            disparity ``x_L - x_R``. ``None`` (default) disables it -- the metric
            measures matchable texture, not geometry.

    Returns:
        ``(K,)`` boolean array, ``True`` where the match is epipolar-consistent.
    """
    left_xy = np.asarray(left_xy, dtype=float).reshape(-1, 2)
    right_xy = np.asarray(right_xy, dtype=float).reshape(-1, 2)
    if left_xy.shape != right_xy.shape:
        raise ValueError(
            f"left_xy and right_xy must share a shape, got {left_xy.shape} vs {right_xy.shape}"
        )

    dy = np.abs(left_xy[:, 1] - right_xy[:, 1])
    mask = dy <= tau

    if disparity_bounds is not None:
        d_min, d_max = disparity_bounds
        disparity = left_xy[:, 0] - right_xy[:, 0]
        mask &= (disparity >= d_min) & (disparity <= d_max)

    return mask
