"""Aggregate sensitivity rows (over videos) and render the markdown summary.

Shared by the live run (`run_sensitivity.py`) and the re-plot script
(`plot_results.py`) so both produce identical aggregates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PSNR is +inf for identical images; clamp so it stays plottable/finite.
PSNR_CAP_DB = 100.0


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """Mean E_match / SSIM / PSNR per (distortion, severity), averaged over videos.

    Severities keep first-seen order (the sweep order), not numeric order, so
    decreasing sweeps (jpeg quality, downscale factor) stay monotone on the x-axis.
    """
    grouped: dict[str, dict] = {}
    for row in rows:
        name = row["distortion"]
        severity = float(row["severity"])
        entry = grouped.setdefault(name, {"trend": row.get("trend", ""), "order": [], "vals": {}})
        if severity not in entry["vals"]:
            entry["order"].append(severity)
            entry["vals"][severity] = {"error": [], "ssim": [], "psnr": []}
        entry["vals"][severity]["error"].append(float(row["error"]))
        entry["vals"][severity]["ssim"].append(float(row["ssim"]))
        entry["vals"][severity]["psnr"].append(min(float(row["psnr"]), PSNR_CAP_DB))

    out: dict[str, dict] = {}
    for name, entry in grouped.items():
        sevs = entry["order"]
        out[name] = {
            "severities": sevs,
            "error": [float(np.mean(entry["vals"][s]["error"])) for s in sevs],
            "ssim": [float(np.mean(entry["vals"][s]["ssim"])) for s in sevs],
            "psnr": [float(np.mean(entry["vals"][s]["psnr"])) for s in sevs],
            "trend": entry["trend"],
        }
    return out


def write_summary(per: dict, path: str | Path, meta: str) -> None:
    """Write a markdown table of E_match / SSIM / PSNR (min severity -> max severity)."""
    lines = [
        "# Matchability sensitivity study",
        "",
        meta,
        "",
        "| distortion | expected | E_match min → max | SSIM min → max | PSNR(dB) min → max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, d in per.items():
        e = [100 * x for x in d["error"]]
        s, p = d["ssim"], d["psnr"]
        lines.append(
            f"| {name} | {d['trend']} | {e[0]:.1f}% → {e[-1]:.1f}% | "
            f"{s[0]:.2f} → {s[-1]:.2f} | {p[0]:.1f} → {p[-1]:.1f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")
