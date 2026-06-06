"""Generate Fig-10-style match overlays for all distortions at a representative severity.

Runs DeDoDe once per distortion on a single video (default: 0001) at the VIZ_TARGET
severity — far faster than the full sweep. Writes to experiments/results/.

Usage:
    python scripts/generate_overlays.py [--video 0001] [--backend dedode]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from matchability import distortions as D
from matchability.imageio_util import load_image
from matchability.metric import _membership_mask
from matchability.types import MatchabilityResult
from matchability.viz import draw_matches

# One representative severity per distortion: severe enough to show a clear mix of
# TP / FP / FN, but not so extreme that everything is lost.
VIZ_TARGETS: dict[str, float] = {
    "brightness_gamma": 2.0,
    "contrast_fade": 0.2,
    "disparity_scale": 1.2,
    "downscale_upscale": 0.25,
    "elastic_warp": 8.0,
    "gaussian_blur": 8.0,
    "gaussian_noise": 40.0,
    "horizontal_shift": 32.0,
    "jpeg": 5.0,
    "occlusion_patch": 0.5,
    "vertical_shift": 6.0,
}


def build_matcher(backend: str, device: str | None):
    if backend == "classical":
        from matchability.matchers.classical import ClassicalMatcher

        return ClassicalMatcher()
    from matchability.matchers.dedode import DeDoDeV2Matcher

    return DeDoDeV2Matcher(device=device)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--video", default=None, help="video stem (default: first available)")
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--backend", default="dedode", choices=["dedode", "classical"])
    p.add_argument("--device", default=None)
    p.add_argument("--working-resolution", type=int, default=768)
    p.add_argument("--n-keypoints", type=int, default=5000)
    p.add_argument("--tau", type=float, default=2.0)
    args = p.parse_args(argv)

    frames_dir = Path(args.frames_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lefts = sorted(frames_dir.glob("*_left.png"))
    if not lefts:
        print("no frames found — run extract_frames.py first")
        return 1

    if args.video:
        stem = args.video
        left_path = frames_dir / f"{stem}_left.png"
        right_path = frames_dir / f"{stem}_right.png"
    else:
        left_path = lefts[0]
        right_path = left_path.with_name(left_path.name.replace("_left", "_right"))
        stem = left_path.name.replace("_left.png", "")

    if not right_path.exists():
        print(f"missing right frame for {stem}")
        return 1

    left = load_image(left_path, args.working_resolution)
    right_gt = load_image(right_path, args.working_resolution)

    matcher = build_matcher(args.backend, args.device)

    # Detect keypoints once, compute GT matches once — reuse for all distortions.
    print(f"detecting keypoints in {stem} ...")
    keypoints = matcher.detect(left, n=args.n_keypoints)
    matches_gt = matcher.match(left, keypoints, right_gt)
    m_gt = _membership_mask(keypoints, matches_gt, args.tau, 0.0, None)

    for name, severity in VIZ_TARGETS.items():
        right_pred = D.apply(name, right_gt, severity)
        matches_pred = matcher.match(left, keypoints, right_pred)
        m_pred = _membership_mask(keypoints, matches_pred, args.tau, 0.0, None)
        res = MatchabilityResult.from_masks(
            m_gt,
            m_pred,
            keypoints_left=keypoints,
            right_xy_gt=matches_gt.right_xy,
            right_xy_pred=matches_pred.right_xy,
        )
        title = (
            f"{name} = {severity}"
            f"  E_match={res.error_pct:.1f}%  TP={res.tp} FP={res.fp} FN={res.fn}"
        )
        overlay = draw_matches(left, right_pred, res, title=title)
        out_path = out / f"matches_{stem}_{name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  {name}: E_match={res.error_pct:.1f}%  TP={res.tp} FP={res.fp} FN={res.fn}")

    print(f"wrote {len(VIZ_TARGETS)} overlays to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
