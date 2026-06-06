"""Visualisation helper: Fig-10-style match overlay for the metric."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from matchability.types import MatchabilityResult

_GREEN = (0, 200, 0)   # true positive (matchable in both)
_ORANGE = (255, 140, 0)  # false positive (hallucination)
_RED = (220, 0, 0)     # false negative (omission)


def draw_matches(
    left: NDArray[np.uint8],
    right: NDArray[np.uint8],
    result: MatchabilityResult,
    *,
    max_per_class: int = 300,
    gap: int = 12,
    title: str | None = None,
    rng_seed: int = 0,
) -> NDArray[np.uint8]:
    """Side-by-side ``left | right`` overlay showing all GT and predicted matches.

    Every matched keypoint from either the GT or predicted view is drawn as a
    coloured line. All three classes are shown simultaneously:

    - 🟢 **Green** (TP) — matched in *both* GT and predicted right.
    - 🟠 **Orange** (FP / hallucination) — in predicted but not GT.
    - 🔴 **Red** (FN / omission) — in GT but lost in predicted; line goes to GT location.

    Up to ``max_per_class`` lines per class (random sample); FN drawn first so red
    is never hidden by green.
    """
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    h, w = left.shape[:2]
    canvas = np.full((h, 2 * w + gap, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = left
    canvas[:h, w + gap : w + gap + w] = right

    kp = result.keypoints_left
    if kp is None or result.mask_gt is None or result.mask_pred is None:
        return canvas

    mask_gt = result.mask_gt
    mask_pred = result.mask_pred
    right_xy_pred = result.right_xy_pred
    right_xy_gt = result.right_xy_gt

    rng = np.random.default_rng(rng_seed)

    def _sample(indices):
        if len(indices) > max_per_class:
            indices = rng.choice(indices, max_per_class, replace=False)
        return indices

    tp_idx = _sample(np.where(mask_gt & mask_pred)[0])
    fp_idx = _sample(np.where(~mask_gt & mask_pred)[0])
    fn_idx = _sample(np.where(mask_gt & ~mask_pred)[0])

    if right_xy_gt is not None:
        for i in fn_idx:
            x, y = int(kp[i][0]), int(kp[i][1])
            rx, ry = int(right_xy_gt[i][0]) + w + gap, int(right_xy_gt[i][1])
            cv2.line(canvas, (x, y), (rx, ry), _RED, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, _RED, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 3, _RED, -1, cv2.LINE_AA)

    if right_xy_pred is not None:
        for i in fp_idx:
            x, y = int(kp[i][0]), int(kp[i][1])
            rx, ry = int(right_xy_pred[i][0]) + w + gap, int(right_xy_pred[i][1])
            cv2.line(canvas, (x, y), (rx, ry), _ORANGE, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, _ORANGE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 3, _ORANGE, -1, cv2.LINE_AA)
        for i in tp_idx:
            x, y = int(kp[i][0]), int(kp[i][1])
            rx, ry = int(right_xy_pred[i][0]) + w + gap, int(right_xy_pred[i][1])
            cv2.line(canvas, (x, y), (rx, ry), _GREEN, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, _GREEN, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 3, _GREEN, -1, cv2.LINE_AA)

    if title:
        cv2.rectangle(canvas, (0, 0), (2 * w + gap, 26), (0, 0, 0), -1)
        cv2.putText(
            canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas
