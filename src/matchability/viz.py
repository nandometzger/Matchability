"""Visualisation helpers: Fig-10-style match overlays and sensitivity plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from matchability.types import MatchabilityResult

_GREEN = (0, 200, 0)  # true positive (matchable in both)
_ORANGE = (255, 140, 0)  # false positive (hallucination)
_RED = (220, 0, 0)  # false negative (omission)


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
    coloured line.  All three classes are shown simultaneously so the viewer can
    see the full picture at a glance:

    - **Green line** (TP) — matched in *both* GT and predicted right.
      Left keypoint → predicted right location.
    - **Orange line** (FP / hallucination) — matched in predicted right but not GT.
      Left keypoint → predicted right location.
    - **Red line** (FN / omission) — matched in GT but lost in the predicted view.
      Left keypoint → GT right location, showing the expected connection that is
      absent in the prediction.

    Up to ``max_per_class`` lines are drawn per class (random sample when there
    are more); drawing order is FN first so red is never hidden by green.
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

    # Draw FN first (red, underneath), then FP, then TP on top
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


_BG_FLAT = "#f0f4ff"   # light blue tint  — insensitive / flat group
_BG_RISES = "#fff4f0"  # light red tint   — sensitive / rises group


def _group_bg(trend: str) -> str:
    return _BG_FLAT if trend == "flat" else _BG_RISES


def plot_distortion_grid(
    per_distortion: Mapping[str, Mapping[str, Sequence]],
    out_path: str | Path,
    *,
    title: str | None = None,
    ncols: int = 4,
) -> Path:
    """Grid of ``E_match`` (%) vs severity per distortion, with an SSIM overlay.

    ``per_distortion[name]`` is a mapping with keys ``severities``, ``error`` and
    optionally ``ssim`` and ``trend``.  Subplots are shaded blue (flat/insensitive)
    or red (rises/sensitive) to make the grouping immediately visible.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = list(per_distortion)
    ncols = max(1, min(ncols, len(names)))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat = axes.flat

    for ax, name in zip(flat, names, strict=False):
        data = per_distortion[name]
        trend = data.get("trend", "")
        ax.set_facecolor(_group_bg(trend))
        x = list(range(len(data["error"])))
        ax.plot(x, [100 * e for e in data["error"]], "o-", color="crimson", label="E_match (%)")
        ax.set_ylim(-2, 102)
        ax.set_ylabel("E_match (%)", color="crimson")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in data["severities"]], rotation=45, fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"{name}  [{trend}]", fontsize=10)
        if data.get("ssim") is not None:
            ax2 = ax.twinx()
            ax2.plot(x, list(data["ssim"]), "s--", color="steelblue", alpha=0.7)
            ax2.set_ylim(0, 1.02)
            ax2.set_ylabel("SSIM", color="steelblue")

    for ax in list(flat)[len(names) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _minmax(values: Sequence[float], lo: float, hi: float) -> list[float]:
    rng = hi - lo
    if rng <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / rng for v in values]


def _degradation(
    values: Sequence[float], lo: float, hi: float, higher_is_worse: bool
) -> list[float]:
    """Min-max scale to [0, 1] and orient so larger == more degradation."""
    norm = _minmax(values, lo, hi)
    return norm if higher_is_worse else [1.0 - x for x in norm]


def plot_metric_comparison_grid(
    per_distortion: Mapping[str, Mapping[str, Sequence]],
    out_path: str | Path,
    *,
    normalize: str = "sweep",
    title: str | None = None,
    ncols: int = 4,
) -> Path:
    """Overlay E_match, SSIM and PSNR per distortion, min-max scaled and degradation-aligned.

    Every metric is normalised to ``[0, 1]`` and oriented so larger means more
    degradation (E_match as-is; SSIM/PSNR flipped to ``1 - norm``), so the three
    curves are directly comparable. ``normalize`` is ``"sweep"`` (each metric over
    its own per-distortion range) or ``"global"`` (over its range across the study).
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = list(per_distortion)
    ranges = None
    if normalize == "global":
        ranges = {}
        for key in ("error", "ssim", "psnr"):
            allv = [v for d in per_distortion.values() for v in d[key]]
            ranges[key] = (min(allv), max(allv))

    ncols = max(1, min(ncols, len(names)))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat = axes.flat

    for ax, name in zip(flat, names, strict=False):
        d = per_distortion[name]
        trend = d.get("trend", "")
        ax.set_facecolor(_group_bg(trend))
        x = list(range(len(d["error"])))
        if ranges is not None:
            re, rs, rp = ranges["error"], ranges["ssim"], ranges["psnr"]
        else:
            re = (min(d["error"]), max(d["error"]))
            rs = (min(d["ssim"]), max(d["ssim"]))
            rp = (min(d["psnr"]), max(d["psnr"]))
        ax.plot(x, _degradation(d["error"], *re, True), "o-", color="crimson", label="E_match")
        ax.plot(x, _degradation(d["ssim"], *rs, False), "s--", color="steelblue", label="1−SSIM")
        ax.plot(x, _degradation(d["psnr"], *rp, False), "^:", color="seagreen", label="1−PSNR")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("degradation (norm.)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in d["severities"]], rotation=45, fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"{name}  [{trend}]", fontsize=10)

    for ax in list(flat)[len(names) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
