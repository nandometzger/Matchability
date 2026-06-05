# Matchability Error — precise specification

This document pins down the operational definition implemented in this package,
including the choices the Elastic3D paper leaves implicit. Where the paper is
explicit we cite the equation; where it is not, the choice is marked
**[implementation choice]** and is configurable.

## Definition (paper)

Given a left image `I_L`, a ground-truth right view `R_gt`, and a predicted
right view `R_pred`, the Matchability Error is

```
E_match = 1 - |M_gt ∩ M_pred| / |M_gt ∪ M_pred|          (Elastic3D eq. 4)
        = (N_FP + N_FN) / (N_TP + N_FP + N_FN)            (Elastic3D eq. 14)
```

where, using a robust matcher (DeDoDe v2):

- `M_gt`   = keypoints in `I_L` that have an **epipolar-consistent** match in `R_gt`,
- `M_pred` = keypoints in `I_L` that have an epipolar-consistent match in `R_pred`,
- `N_TP = |M_gt ∩ M_pred|` (correct, matchable detail),
- `N_FP = |M_pred \ M_gt|`  (**hallucination** — detail matchable in the prediction but not the GT),
- `N_FN = |M_gt \ M_pred|`  (**omission** — detail matchable in the GT but lost in the prediction).

The result is reported ×100 (a percentage), matching the paper's tables.
A lower error means the synthesized view preserves consistent, matchable texture
along the correct epipolar geometry — i.e. less binocular rivalry.

Sanity check against Fig. 10 (paper): TP=4717, FP=1908, FN=1354 →
`(1908+1354)/(4717+1908+1354) = 0.409` → `E_match = 40.9`. ✓

## Making the two sets index-comparable [implementation choice]

`M_gt` and `M_pred` are subsets of keypoints **in the same image `I_L`**, so the
Jaccard index needs a shared reference set. We:

1. Detect **one fixed reference keypoint set** `K_L` in `I_L` (detected once).
2. Match `K_L` into `R_gt` and into `R_pred` independently.
3. `M_gt` / `M_pred` are **boolean membership masks over `K_L`** (a reference
   keypoint is "in" the set iff it has an epipolar-consistent, confident match).

This is the only well-defined reading of eq. 4 (intersection/union require a
common universe) and reproduces Fig. 10, where the True-Positive count is shared
across the FP and FN panels. The decomposition then reduces to boolean algebra
over the `K` reference keypoints:

```
TP = (M_gt & M_pred).sum()
FP = (~M_gt & M_pred).sum()
FN = (M_gt & ~M_pred).sum()
```

## Epipolar consistency [implementation choice]

The inputs are **rectified** stereo (verified on the AVP data: median vertical
disparity ≈ 1px at 2200px, sub-pixel at the working resolution). For rectified
pairs the epipolar lines are horizontal rows, so a match `(x_L,y_L)→(x_R,y_R)` is
epipolar-consistent iff

```
|y_L - y_R| <= tau          # default tau = 2 px at the working resolution
```

An optional non-negative-disparity bound (`x_L - x_R` within `[d_min, d_max]`) is
available but **off by default** — the metric measures *matchable texture*, not
geometry (geometry is the paper's separate `Disp. err`). A general fundamental-
matrix / Sampson-distance mode is provided for non-rectified inputs.

## Defaults

| Knob                  | Default            | Notes |
| --------------------- | ------------------ | ----- |
| matcher               | DeDoDe v2          | kornia `detector=L-C4-v2`, `descriptor=G-upright` |
| working resolution    | 768 (long side)    | inputs are resized; keypoints/`tau` live in this space |
| `n_keypoints`         | 5000               | size of the fixed reference set `K_L` |
| `tau` (epipolar)      | 2 px               | at the working resolution |
| match confidence      | backend default    | mutual-NN + similarity threshold |
| device                | auto: mps>cuda>cpu | MPS uses float32 + `PYTORCH_ENABLE_MPS_FALLBACK=1` |

All are exposed via `Matchability(...)`. The absolute counts depend on these
knobs, but `E_match` is a *ratio* and is comparatively stable to `n_keypoints`.

## Expected behavior (paper App. D.1 — used as test assertions)

- **Gaussian blur → error rises sharply** (over-smoothing destroys keypoints).
- **Horizontal shift → error stays flat** (DeDoDe is translation-invariant; this
  decouples texture fidelity from geometry).
- Headline: at matched severities, `E_match(blur) ≫ E_match(horizontal shift)`.

This package additionally checks: identity → ≈0, unrelated image → ≈1, vertical
shift → rises (breaks epipolar), occlusion → rises with area, brightness/gamma →
≈flat (descriptor robustness).
