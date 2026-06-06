"""VAE encoding-decoding sensitivity experiment — 5 VAE architectures.

Compares how five different VAE models degrade the Matchability metric when used
to encode-decode the right view. All models are publicly available on HuggingFace,
require no authentication, and run on Apple Silicon MPS.

VAEs tested (roughly worst → best reconstruction quality):
  1. TAESD       madebyollin/taesd           ~4 MB   SD1 tiny autoencoder
  2. TAESDXL     madebyollin/taesdxl         ~5 MB   SDXL tiny autoencoder
  3. SD-VAE-MSE  stabilityai/sd-vae-ft-mse   ~80 MB  SD1 full VAE (MSE fine-tune)
  4. SD-VAE-EMA  stabilityai/sd-vae-ft-ema   ~80 MB  SD1 full VAE (EMA fine-tune)
  5. SDXL-VAE    madebyollin/sdxl-vae-fp16-fix ~168 MB SDXL full VAE

Results are written to experiments/results_vae/:
  vae_sensitivity.csv    — per-pair, per-VAE: E_match, SSIM, PSNR
  vae_comparison.png     — E_match / 1-SSIM / 1-PSNR across VAEs
  vae_summary.md         — markdown summary table
  vae_overlays/          — match overlays for each VAE

Usage:
    pip install -e ".[dev,viz]"
    pip install diffusers accelerate
    python scripts/run_vae_experiment.py [--backend dedode|classical]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from matchability.imageio_util import load_image
from matchability.metric import _membership_mask
from matchability.quality import psnr, ssim
from matchability.report import aggregate, write_summary
from matchability.types import MatchabilityResult
from matchability.viz import draw_matches


def _plot_vae_bar(per: dict, out_path, *, n_pairs: int) -> None:
    """Bar chart of E_match (%) across VAE architectures."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    names = list(per)
    ematch = [per[n]["error"][0] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2), 4))
    bars = ax.bar(names, ematch, color="crimson", alpha=0.85, edgecolor="darkred", linewidth=0.8)
    for bar, val in zip(bars, ematch, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("E_match (%)", color="crimson")
    ax.set_ylim(0, max(ematch) * 1.25 + 2)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Matchability Error — VAE roundtrip ({n_pairs} pairs, DeDoDe v2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

# Five VAEs ordered from worst to best expected reconstruction quality.
# Each entry: (display_name, hf_repo, loader_cls_name, severity_index)
VAES: list[tuple[str, str, str, float]] = [
    ("TAESD", "madebyollin/taesd", "AutoencoderTiny", 1.0),
    ("TAESDXL", "madebyollin/taesdxl", "AutoencoderTiny", 2.0),
    ("SD-VAE-MSE", "stabilityai/sd-vae-ft-mse", "AutoencoderKL", 3.0),
    ("SD-VAE-EMA", "stabilityai/sd-vae-ft-ema", "AutoencoderKL", 4.0),
    ("SDXL-VAE", "madebyollin/sdxl-vae-fp16-fix", "AutoencoderKL", 5.0),
]

_MODEL_CACHE: dict[str, object] = {}


def _load_vae(repo: str, cls_name: str, device: torch.device):
    key = f"{repo}@{device}"
    if key not in _MODEL_CACHE:
        import diffusers

        cls = getattr(diffusers, cls_name)
        model = cls.from_pretrained(repo)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _pad16(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    ph = ((h + 15) // 16) * 16
    pw = ((w + 15) // 16) * 16
    if ph != h or pw != w:
        image = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    return image, h, w


def vae_roundtrip(
    image: np.ndarray,
    repo: str,
    cls_name: str,
    device: torch.device,
) -> np.ndarray:
    """Encode image with the given VAE and decode back to pixel space."""
    model = _load_vae(repo, cls_name, device)
    padded, oh, ow = _pad16(image)
    tensor = torch.from_numpy(padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        encoded = model.encode(tensor)
        # AutoencoderTiny returns .latents directly;
        # AutoencoderKL returns .latent_dist (a diagonal Gaussian — use mode for determinism).
        if hasattr(encoded, "latents"):
            latent = encoded.latents
        else:
            latent = encoded.latent_dist.mode()
        decoded = model.decode(latent).sample
    out = decoded.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    out = (out.cpu().numpy() * 255).astype(np.uint8)
    return out[:oh, :ow]


def build_matcher(backend: str, device_str: str | None):
    if backend == "classical":
        from matchability.matchers.classical import ClassicalMatcher

        return ClassicalMatcher()
    from matchability.matchers.dedode import DeDoDeV2Matcher

    return DeDoDeV2Matcher(device=device_str)


def load_pairs(frames_dir: Path, resolution: int):
    pairs = []
    for lp in sorted(frames_dir.glob("*_left.png")):
        rp = lp.with_name(lp.name.replace("_left", "_right"))
        if rp.exists():
            stem = lp.name.replace("_left.png", "")
            pairs.append((stem, load_image(lp, resolution), load_image(rp, resolution)))
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", default="data/frames")
    p.add_argument("--out", default="experiments/results_vae")
    p.add_argument("--backend", default="dedode", choices=["dedode", "classical"])
    p.add_argument("--device", default=None)
    p.add_argument("--working-resolution", type=int, default=768)
    p.add_argument("--n-keypoints", type=int, default=5000)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--viz-video", default=None)
    args = p.parse_args(argv)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overlays_dir = out / "vae_overlays"
    overlays_dir.mkdir(exist_ok=True)

    pairs = load_pairs(Path(args.frames_dir), args.working_resolution)
    if not pairs:
        print("no stereo pairs found — run extract_frames.py first")
        return 1

    matcher = build_matcher(args.backend, args.device)
    viz_stem = args.viz_video or pairs[0][0]

    if args.device:
        vae_device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        vae_device = torch.device("mps")
    else:
        vae_device = torch.device("cpu")
    print(f"VAE device: {vae_device}  |  matcher: {args.backend}")

    # Pre-load all VAEs
    for display_name, repo, cls_name, _ in VAES:
        print(f"loading {display_name} ({repo}) ...")
        _load_vae(repo, cls_name, vae_device)

    rows: list[dict] = []

    for stem, left, right_gt in pairs:
        keypoints = matcher.detect(left, n=args.n_keypoints)
        matches_gt = matcher.match(left, keypoints, right_gt)
        m_gt = _membership_mask(keypoints, matches_gt, args.tau, 0.0, None)

        for display_name, repo, cls_name, sev_idx in VAES:
            right_pred = vae_roundtrip(right_gt, repo, cls_name, vae_device)
            matches_pred = matcher.match(left, keypoints, right_pred)
            m_pred = _membership_mask(keypoints, matches_pred, args.tau, 0.0, None)
            res = MatchabilityResult.from_masks(
                m_gt,
                m_pred,
                keypoints_left=keypoints,
                right_xy_gt=matches_gt.right_xy,
                right_xy_pred=matches_pred.right_xy,
            )
            quality_ssim = ssim(right_gt, right_pred)
            quality_psnr = psnr(right_gt, right_pred)
            rows.append(
                {
                    "video": stem,
                    "distortion": display_name,
                    "trend": "rises",
                    "severity": sev_idx,
                    "error": res.error,
                    "error_pct": res.error_pct,
                    "tp": res.tp,
                    "fp": res.fp,
                    "fn": res.fn,
                    "ssim": quality_ssim,
                    "psnr": quality_psnr,
                }
            )

            if stem == viz_stem:
                title = (
                    f"{display_name}  E_match={res.error_pct:.1f}%"
                    f"  TP={res.tp} FP={res.fp} FN={res.fn}"
                )
                overlay = draw_matches(left, right_pred, res, title=title)
                cv2.imwrite(
                    str(overlays_dir / f"matches_{stem}_{display_name}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )

        print(f"[{stem}] all VAEs done")

    with open(out / "vae_sensitivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per = aggregate(rows)
    _plot_vae_bar(per, out / "vae_comparison.png", n_pairs=len(pairs))

    meta = (
        "| VAE | repo | size |\n"
        "| --- | --- | --- |\n"
        + "\n".join(
            f"| {n} | `{r}` | AutoencoderTiny/KL |"
            for n, r, _, _ in VAES
        )
        + "\n\n"
        f"- backend: `{args.backend}`  ·  pairs: {len(pairs)}"
        f"  ·  resolution: {args.working_resolution}px"
        f"  ·  keypoints: {args.n_keypoints}  ·  tau: {args.tau}px"
    )
    write_summary(per, out / "vae_summary.md", meta)
    print(f"wrote results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
