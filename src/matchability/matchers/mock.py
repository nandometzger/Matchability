"""A deterministic, programmable matcher for unit tests (no model, no pixels)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from matchability.matchers.base import Matcher
from matchability.types import LeftToRightMatches


def _default_key(image: NDArray) -> int:
    """Key an image by its first element -- handy for tagging test fixtures."""
    return int(np.asarray(image).reshape(-1)[0])


class MockMatcher(Matcher):
    """Returns a fixed reference keypoint set and pre-registered matches per right image.

    Right images are identified via ``key_fn`` (by default their first pixel value),
    letting tests prescribe exactly which correspondences come back for the GT and
    predicted views.
    """

    def __init__(
        self,
        keypoints: NDArray[np.floating],
        responses: Mapping[int, LeftToRightMatches],
        key_fn: Callable[[NDArray], int] | None = None,
    ) -> None:
        self._keypoints = np.asarray(keypoints, dtype=float).reshape(-1, 2)
        self._responses = dict(responses)
        self._key_fn = key_fn or _default_key

    def detect(self, image: NDArray, n: int | None = None) -> NDArray[np.floating]:
        kp = self._keypoints.copy()
        return kp if n is None else kp[:n]

    def match(
        self, image_left: NDArray, keypoints_left: NDArray, image_right: NDArray
    ) -> LeftToRightMatches:
        return self._responses[self._key_fn(image_right)]
