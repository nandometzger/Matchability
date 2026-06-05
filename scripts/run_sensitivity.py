"""Empirical sensitivity study of the Matchability metric on real stereo pairs.

For every stereo pair we fix the GT side once, then sweep each distortion's
severities on the right view (R_pred = distort(R_gt)) and record E_match (with its
TP/FP/FN decomposition) alongside SSIM/PSNR. Outputs a CSV, a per-distortion plot
grid, a markdown summary, and a few Fig-10-style match overlays.

Usage:
    python scripts/run_sensitivity.py --backend dedode --out experiments/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from matchability import distortions as D
from matchability.imageio_util import load_image
from matchability.matchers.base import Matcher
from matchability.metric import _membership_mask
from matchability.quality import psnr, ssim
from matchability.types import MatchabilityResult
from matchability.viz import draw_matches, plot_distortion_grid

# Representative (distortion, severity) cases rendered as match overlays.
VIZ_TARGETS = {
    "identity": 0.0,
    "gaussian_blur": 8.0,
    "horizontal_shift": 32.0,
    "vertical_shift": 6.0,
    "occlusion_patch": 0.5,
}


def build_matcher(backend: str, device: str | None) -> Matcher:
    if backend == "classical":
        from matchability.matchers.classical import ClassicalMatcher

        return ClassicalMatcher()
    from matchability.matchers.dedode import DeDoDeV2Matcher

    return DeDoDeV2Matcher(device=device)


def load_pairs(frames_dir: Path, videos_dir: Path, resolution: int, frame_index: int):
    pairs = []
    lefts = sorted(frames_dir.glob("*_left.png"))
    if lefts:
        for lp in lefts:
            rp = lp.with_name(lp.name.replace("_left", "_right"))
            if rp.exists():
                stem = lp.name.replace("_left.png", "")
                pairs.append((stem, load_image(lp, resolution), load_image(rp, resolution)))
        return pairs
    from matchability.io_video import extract_stereo_frame

    for video in sorted(videos_dir.glob("*.mov")):
        left, right = extract_stereo_frame(video, frame_index)
        pairs.append((video.stem, load_image(left, resolution), load_image(right, resolution)))
    return pairs


def write_summary(path: Path, per: dict, n_pairs: int, args) -> None:
    lines = [
        "# Matchability sensitivity study",
        "",
        f"- backend: `{args.backend}`  ·  pairs: {n_pairs}"
        f"  ·  resolution: {args.working_resolution}px  ·  keypoints: {args.n_keypoints}"
        f"  ·  tau: {args.tau}px",
        "- `E_match` averaged over all pairs, reported as a percentage.",
        "",
        "| distortion | expected | E_match @ min sev | E_match @ max sev | SSIM min→max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, d in per.items():
        e = [100 * x for x in d["error"]]
        s = d["ssim"]
        lines.append(
            f"| {name} | {d['trend']} | {e[0]:.1f}% (sev {d['severities'][0]}) | "
            f"{e[-1]:.1f}% (sev {d['severities'][-1]}) | {s[0]:.2f}→{s[-1]:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--videos-dir", default="data/raw")
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--backend", default="dedode", choices=["dedode", "classical"])
    p.add_argument("--device", default=None)
    p.add_argument("--working-resolution", type=int, default=768)
    p.add_argument("--n-keypoints", type=int, default=5000)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--conf-threshold", type=float, default=0.0)
    p.add_argument("--distortions", nargs="*", default=None, help="subset (default: all)")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--viz-video", default=None)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(
        Path(args.frames_dir), Path(args.videos_dir), args.working_resolution, args.frame_index
    )
    if not pairs:
        print("no stereo pairs found (run extract_frames.py first)")
        return 1
    names = args.distortions or D.available()
    matcher = build_matcher(args.backend, args.device)
    viz_stem = args.viz_video or pairs[0][0]

    rows: list[dict] = []
    agg: dict[str, dict[float, dict[str, list]]] = {}

    for stem, left, right_gt in pairs:
        keypoints = matcher.detect(left, n=args.n_keypoints)
        matches_gt = matcher.match(left, keypoints, right_gt)
        m_gt = _membership_mask(keypoints, matches_gt, args.tau, args.conf_threshold, None)
        for name in names:
            dist = D.get(name)
            for severity in dist.severities:
                right_pred = D.apply(name, right_gt, severity)
                matches_pred = matcher.match(left, keypoints, right_pred)
                m_pred = _membership_mask(
                    keypoints, matches_pred, args.tau, args.conf_threshold, None
                )
                res = MatchabilityResult.from_masks(
                    m_gt,
                    m_pred,
                    keypoints_left=keypoints,
                    right_xy_gt=matches_gt.right_xy,
                    right_xy_pred=matches_pred.right_xy,
                )
                quality_ssim = ssim(right_gt, right_pred)
                rows.append(
                    {
                        "video": stem,
                        "distortion": name,
                        "trend": dist.trend,
                        "severity": severity,
                        "error": res.error,
                        "error_pct": res.error_pct,
                        "tp": res.tp,
                        "fp": res.fp,
                        "fn": res.fn,
                        "ssim": quality_ssim,
                        "psnr": psnr(right_gt, right_pred),
                    }
                )
                agg.setdefault(name, {}).setdefault(severity, {"err": [], "ssim": []})
                agg[name][severity]["err"].append(res.error)
                agg[name][severity]["ssim"].append(quality_ssim)
                if stem == viz_stem and VIZ_TARGETS.get(name) == severity:
                    title = (
                        f"{name}={severity}  E_match={res.error_pct:.1f}%  "
                        f"TP={res.tp} FP={res.fp} FN={res.fn}"
                    )
                    overlay = draw_matches(left, right_pred, res, title=title)
                    cv2.imwrite(
                        str(out / f"matches_{stem}_{name}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
            print(f"[{stem}] {name}: done")

    with open(out / "sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per = {}
    for name in names:
        sevs = list(D.get(name).severities)
        per[name] = {
            "severities": sevs,
            "error": [float(np.mean(agg[name][s]["err"])) for s in sevs],
            "ssim": [float(np.mean(agg[name][s]["ssim"])) for s in sevs],
            "trend": D.get(name).trend,
        }
    grid_title = f"Matchability sensitivity ({args.backend}, {len(pairs)} pairs)"
    plot_distortion_grid(per, out / "sensitivity_grid.png", title=grid_title)
    write_summary(out / "summary.md", per, len(pairs), args)
    print(f"wrote results to {out}/ (csv, grid png, summary.md, match overlays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
