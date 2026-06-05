"""Command-line interface: ``matchability LEFT RIGHT_GT RIGHT_PRED``."""

from __future__ import annotations

import argparse

from matchability.api import Matchability
from matchability.matchers.base import Matcher


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matchability",
        description="Compute the Elastic3D Matchability Error for a stereo triplet.",
    )
    p.add_argument("left", help="left image")
    p.add_argument("right_gt", help="ground-truth right image")
    p.add_argument("right_pred", help="predicted right image")
    p.add_argument(
        "--backend",
        choices=["dedode", "classical"],
        default="dedode",
        help="matcher backend (default: dedode v2)",
    )
    p.add_argument("--tau", type=float, default=2.0, help="epipolar threshold in pixels")
    p.add_argument("--n-keypoints", type=int, default=5000)
    p.add_argument("--working-resolution", type=int, default=768)
    p.add_argument("--conf-threshold", type=float, default=0.0)
    p.add_argument("--device", default=None, help="mps/cuda/cpu (default: auto)")
    p.add_argument("--viz", default=None, help="write a match-classification PNG to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    matcher: Matcher | None = None
    if args.backend == "classical":
        from matchability.matchers.classical import ClassicalMatcher

        matcher = ClassicalMatcher()

    metric = Matchability(
        matcher=matcher,
        tau=args.tau,
        n_keypoints=args.n_keypoints,
        working_resolution=args.working_resolution,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )
    result = metric(args.left, args.right_gt, args.right_pred)

    print(f"E_match = {result.error_pct:.2f}%  (error={result.error:.4f})")
    print(
        f"TP={result.tp}  FP={result.fp} (hallucination)  "
        f"FN={result.fn} (omission)  n_ref={result.n_reference}"
    )

    if args.viz:
        import cv2

        from matchability.imageio_util import load_image
        from matchability.viz import draw_matches

        left = load_image(args.left, args.working_resolution)
        right_pred = load_image(args.right_pred, args.working_resolution)
        title = f"E_match={result.error_pct:.1f}%  TP={result.tp} FP={result.fp} FN={result.fn}"
        overlay = draw_matches(left, right_pred, result, title=title)
        cv2.imwrite(args.viz, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.viz}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
