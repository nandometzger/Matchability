"""Re-plot all results from an existing sensitivity.csv without re-running DeDoDe.

Reads ``experiments/results/sensitivity.csv``, aggregates over videos, then writes:
  - ``sensitivity_grid.png``           — E_match + SSIM per distortion
  - ``comparison_sweep.png``           — E_match / 1-SSIM / 1-PSNR (per-sweep normalised)
  - ``comparison_global.png``          — same, globally normalised
  - ``summary.md``                     — markdown table with PSNR column

For match overlays (require the actual image data and DeDoDe), run
``run_sensitivity.py --backend dedode``.  This script generates everything
derivable from the CSV alone.

Usage:
    python scripts/plot_results.py [--csv experiments/results/sensitivity.csv]
                                   [--out experiments/results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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

    from matchability import distortions as D
    from matchability.report import aggregate, write_summary
    from matchability.viz import plot_distortion_grid, plot_metric_comparison_grid

    rows = load_csv(csv_path)
    # Drop anchor distortions (identity / scramble) — single-point, PSNR=∞ degenerate cases.
    sweep_names = set(D.sweep_distortions())
    rows = [r for r in rows if r["distortion"] in sweep_names]

    # Parse numeric fields — CSV stores everything as strings.
    for row in rows:
        row["severity"] = float(row["severity"])
        row["error"] = float(row["error"])
        row["ssim"] = float(row["ssim"])
        row["psnr"] = float(row["psnr"])

    videos = sorted({r["video"] for r in rows})
    distortions = list(dict.fromkeys(r["distortion"] for r in rows))  # preserve order
    n_pairs = len(videos)
    print(f"loaded {len(rows)} rows · {n_pairs} videos · {len(distortions)} distortions")

    per = aggregate(rows)

    # 1) E_match + SSIM grid (original single-metric view)
    grid_path = plot_distortion_grid(
        per,
        out / "sensitivity_grid.png",
        title=f"Matchability sensitivity ({n_pairs} pairs)",
    )
    print(f"wrote {grid_path}")

    # 2) Three-metric comparison grid (per-sweep normalised)
    cmp_path = plot_metric_comparison_grid(
        per,
        out / "comparison.png",
        normalize="sweep",
        title=f"E_match / 1−SSIM / 1−PSNR  (per-sweep normalised, {n_pairs} pairs)",
    )
    print(f"wrote {cmp_path}")

    # 3) Markdown summary with PSNR column
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
