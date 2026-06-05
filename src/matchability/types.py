"""Result types for the Matchability metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LeftToRightMatches:
    """Per-reference-keypoint correspondence from the left image into a right view.

    All arrays are aligned to the fixed left reference keypoints (length ``K``).

    Attributes:
        right_xy: ``(K, 2)`` matched right ``(x, y)`` coordinate per reference keypoint.
            Entries where ``valid`` is ``False`` are ignored.
        confidence: ``(K,)`` match confidence/score per reference keypoint.
        valid: ``(K,)`` whether a correspondence was found for the reference keypoint.
    """

    right_xy: NDArray[np.floating]
    confidence: NDArray[np.floating]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        right_xy = np.asarray(self.right_xy, dtype=float).reshape(-1, 2)
        confidence = np.asarray(self.confidence, dtype=float).reshape(-1)
        valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        if not (len(right_xy) == len(confidence) == len(valid)):
            raise ValueError(
                "right_xy, confidence and valid must have the same length, got "
                f"{len(right_xy)}, {len(confidence)}, {len(valid)}"
            )
        object.__setattr__(self, "right_xy", right_xy)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)

    def __len__(self) -> int:
        return len(self.valid)


@dataclass(frozen=True)
class MatchabilityResult:
    """Outcome of a Matchability Error computation.

    The metric compares two boolean membership masks over a *shared* set of
    reference keypoints detected in the left image: ``M_gt`` (a keypoint has an
    epipolar-consistent match in the GT right view) and ``M_pred`` (likewise in
    the predicted right view).

    Attributes:
        tp: ``|M_gt & M_pred|`` -- correct, matchable detail.
        fp: ``|~M_gt & M_pred|`` -- hallucination (matchable in pred, not GT).
        fn: ``|M_gt & ~M_pred|`` -- omission (matchable in GT, not pred).
        n_reference: number of reference keypoints in the left image.
    """

    tp: int
    fp: int
    fn: int
    n_reference: int

    @classmethod
    def from_masks(
        cls, m_gt: NDArray[np.bool_], m_pred: NDArray[np.bool_]
    ) -> MatchabilityResult:
        """Build a result from two boolean masks over the same reference keypoints."""
        m_gt = np.asarray(m_gt, dtype=bool)
        m_pred = np.asarray(m_pred, dtype=bool)
        if m_gt.shape != m_pred.shape:
            raise ValueError(
                f"m_gt and m_pred must share a shape, got {m_gt.shape} vs {m_pred.shape}"
            )
        tp = int(np.count_nonzero(m_gt & m_pred))
        fp = int(np.count_nonzero(~m_gt & m_pred))
        fn = int(np.count_nonzero(m_gt & ~m_pred))
        return cls(tp=tp, fp=fp, fn=fn, n_reference=int(m_gt.size))

    @property
    def union(self) -> int:
        """``|M_gt ∪ M_pred| = TP + FP + FN``."""
        return self.tp + self.fp + self.fn

    @property
    def error(self) -> float:
        """Matchability Error as a fraction in ``[0, 1]`` (eq. 4 / eq. 14).

        ``(FP + FN) / (TP + FP + FN)``. By convention the empty-union case
        (no matchable keypoints in either view) is an error of ``0.0``.
        """
        u = self.union
        return 0.0 if u == 0 else (self.fp + self.fn) / u

    @property
    def error_pct(self) -> float:
        """Matchability Error scaled to a percentage, matching the paper's tables."""
        return 100.0 * self.error

    @property
    def jaccard(self) -> float:
        """Jaccard index ``TP / (TP + FP + FN)`` (``1.0`` for empty union)."""
        u = self.union
        return 1.0 if u == 0 else self.tp / u
