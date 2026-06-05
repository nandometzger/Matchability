"""Abstract matcher interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from matchability.types import LeftToRightMatches


class Matcher(ABC):
    """Detect a fixed reference keypoint set in the left image and match it right.

    Implementations must be deterministic for a given input so the metric is
    reproducible. Images are canonical ``(H, W, 3)`` uint8 RGB arrays.
    """

    @abstractmethod
    def detect(self, image: NDArray[np.uint8], n: int | None = None) -> NDArray[np.floating]:
        """Return up to ``n`` reference keypoints as an ``(K, 2)`` array of ``(x, y)``."""

    @abstractmethod
    def match(
        self,
        image_left: NDArray[np.uint8],
        keypoints_left: NDArray[np.floating],
        image_right: NDArray[np.uint8],
    ) -> LeftToRightMatches:
        """Match the given left keypoints into ``image_right`` (aligned to ``keypoints_left``)."""
