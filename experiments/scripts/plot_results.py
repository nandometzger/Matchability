"""Re-plot all results from an existing sensitivity.csv without re-running DeDoDe.

Reads ``experiments/results/sensitivity.csv``, aggregates over videos, then writes:
  - ``comparison.png``   — E_match / 1-SSIM / 1-PSNR (per-sweep normalised)
  - ``summary.md``       — markdown table with PSNR column

Panels are ordered flat (insensitive) first, then rises (sensitive).
Identity-equivalent starting severities (0.0 or the no-op value for each
distortion) are stripped before plotting so curves start at genuine degradation.

Usage:
    python scripts/plot_results.py [--csv experiments/results/sensitivity.csv]
                                   [--out experiments/results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Severity values that are equivalent to identity (R_pred == R_gt) for each distortion.
# These appear as the first point in the CSV from the old sweep and must be stripped.
_IDENTITY_SEVERITY: dict[str, float] = {
    "contrast_fade": 1.0,       # multiply by 1.0 = no change
    "downscale_upscale": 1.0,   # scale 1.0 = no resize
    "disparity_scale": 1.0,     # scale 1.0 = no stretch
    "horizontal_shift": 1.0,    # 1 px shift is imperceptible
    # six distortions previously had severity=0.0 as identity (covered below)
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="experiments/results/sensitivity.csv")
    p.add_argument("--out", default="experiments/results")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run  python scripts/run_sensitivity.py  first to generate it.")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from experiments.lib.plots import plot_metric_comparison_grid
    from experiments.lib.report import aggregate, write_summary
    from matchability import distortions as D

    rows = load_csv(csv_path)

    # Parse numeric fields first so comparisons work.
    for row in rows:
        row["severity"] = float(row["severity"])
        row["error"] = float(row["error"])
        row["ssim"] = float(row["ssim"])
        row["psnr"] = float(row["psnr"])

    # Drop anchor distortions (identity / scramble).
    sweep_names = set(D.sweep_distortions())
    rows = [r for r in rows if r["distortion"] in sweep_names]
    # Drop global severity=0.0 (degenerate identity from old sweeps).
    rows = [r for r in rows if r["severity"] != 0.0]
    # Drop per-distortion identity severities.
    rows = [
        r for r in rows
        if not (
            r["distortion"] in _IDENTITY_SEVERITY
            and r["severity"] == _IDENTITY_SEVERITY[r["distortion"]]
        )
    ]

    videos = sorted({r["video"] for r in rows})
    n_pairs = len(videos)
    print(f"after filtering: {len(rows)} rows · {n_pairs} videos")

    per_raw = aggregate(rows)

    # Reorder: flat (insensitive) distortions first, then rises (sensitive).
    ordered = D.sweep_distortions()  # already flat-first
    per = {n: per_raw[n] for n in ordered if n in per_raw}

    # Comparison grid — flat-first, ncols=3 so all flat fill one row
    cmp_path = plot_metric_comparison_grid(
        per,
        out / "comparison.png",
        normalize="sweep",
        title=f"E_match / 1−SSIM / 1−PSNR  (per-sweep normalised, {n_pairs} pairs)",
        ncols=3,
    )
    print(f"wrote {cmp_path}")

    meta = (
        f"- pairs: {n_pairs}  ·  videos: {', '.join(videos)}\n"
        "- `R_pred = distort(R_gt)` for each severity; DeDoDe v2 on MPS, 768 px, 5000 kpts."
    )
    summary_path = out / "summary.md"
    write_summary(per, summary_path, meta)
    print(f"wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
