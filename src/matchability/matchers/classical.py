"""A classical SIFT + mutual-NN matcher backend.

Used for the fast property tests and CI (no model weights, no GPU). Keypoint
detection is deterministic, so re-detecting the left image inside ``match`` yields
descriptors aligned to the fixed reference set returned by ``detect``.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from matchability.matchers.base import Matcher
from matchability.types import LeftToRightMatches


class ClassicalMatcher(Matcher):
    """SIFT detector/descriptor matched with a Lowe-ratio mutual nearest neighbour."""

    def __init__(self, n_features: int = 8000, ratio: float = 0.8, mutual: bool = True) -> None:
        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        self._ratio = ratio
        self._mutual = mutual

    def _detect_and_describe(
        self, image: NDArray[np.uint8], n: int | None
    ) -> tuple[NDArray[np.floating], NDArray[np.float32]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 128), dtype=np.float32)
        order = np.argsort([-kp.response for kp in keypoints])  # strongest first
        if n is not None:
            order = order[:n]
        xy = np.array([keypoints[i].pt for i in order], dtype=float)
        return xy, descriptors[order].astype(np.float32)

    def detect(self, image: NDArray[np.uint8], n: int | None = None) -> NDArray[np.floating]:
        xy, _ = self._detect_and_describe(image, n)
        return xy

    def match(
        self,
        image_left: NDArray[np.uint8],
        keypoints_left: NDArray[np.floating],
        image_right: NDArray[np.uint8],
    ) -> LeftToRightMatches:
        k = len(keypoints_left)
        right_xy = np.full((k, 2), np.nan, dtype=float)
        confidence = np.zeros(k, dtype=float)
        valid = np.zeros(k, dtype=bool)
        if k == 0:
            return LeftToRightMatches(right_xy, confidence, valid)

        _, desc_left = self._detect_and_describe(image_left, k)
        xy_right, desc_right = self._detect_and_describe(image_right, None)
        if len(desc_left) == 0 or len(desc_right) < 1:
            return LeftToRightMatches(right_xy, confidence, valid)

        knn = self._bf.knnMatch(desc_left, desc_right, k=2)
        reverse_best: dict[int, int] = {}
        if self._mutual:
            for m in self._bf.match(desc_right, desc_left):
                reverse_best[m.queryIdx] = m.trainIdx

        for i, pair in enumerate(knn):
            if not pair:
                continue
            best = pair[0]
            if len(pair) >= 2:
                second = pair[1]
                if best.distance >= self._ratio * second.distance:
                    continue
                score = max(0.0, 1.0 - best.distance / (second.distance + 1e-9))
            else:
                score = 1.0
            j = best.trainIdx
            if self._mutual and reverse_best.get(j, -1) != i:
                continue
            valid[i] = True
            right_xy[i] = xy_right[j]
            confidence[i] = score

        return LeftToRightMatches(right_xy, confidence, valid)
