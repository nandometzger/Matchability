"""A registry of severity-parameterised image distortions for probing the metric.

Each distortion maps a canonical ``(H, W, 3)`` uint8 RGB image and a scalar
severity to a distorted image of the same shape/dtype. ``severities`` are ordered
from least to most degradation. ``trend`` records the *expected* effect on the
Matchability Error and is asserted in the property tests:

* ``rises``       -- error increases with severity (texture loss / epipolar break)
* ``flat``        -- error stays ~constant (matcher is invariant to this change)
* ``anchor_low``  -- error ~0 (no change)
* ``anchor_high`` -- error ~1 (correspondence destroyed)

Randomised distortions use a fixed seed so they are fully deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
_SEED = 0


@dataclass(frozen=True)
class Distortion:
    """A named, severity-parameterised distortion with its expected metric trend."""

    name: str
    fn: Callable[[Image, float], Image]
    severities: tuple[float, ...]
    trend: str
    family: str

    def __call__(self, image: Image, severity: float) -> Image:
        out = self.fn(image, severity)
        return np.ascontiguousarray(out, dtype=np.uint8)


REGISTRY: dict[str, Distortion] = {}


def register(
    name: str,
    fn: Callable[[Image, float], Image],
    severities: tuple[float, ...],
    trend: str,
    family: str,
) -> None:
    REGISTRY[name] = Distortion(name, fn, severities, trend, family)


def available() -> list[str]:
    """Sorted names of registered distortions."""
    return sorted(REGISTRY)


def get(name: str) -> Distortion:
    return REGISTRY[name]


def apply(name: str, image: Image, severity: float) -> Image:
    """Apply a registered distortion by name."""
    return REGISTRY[name](image, severity)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shift(image: Image, dx: float, dy: float) -> Image:
    h, w = image.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------- #
# distortion functions
# --------------------------------------------------------------------------- #
def _identity(image: Image, severity: float) -> Image:
    return image.copy()


def _gaussian_blur(image: Image, sigma: float) -> Image:
    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma))


def _horizontal_shift(image: Image, px: float) -> Image:
    return _shift(image, float(px), 0.0)


def _vertical_shift(image: Image, px: float) -> Image:
    return _shift(image, 0.0, float(px))


def _gaussian_noise(image: Image, std: float) -> Image:
    rng = np.random.default_rng(_SEED)
    noise = rng.normal(0.0, float(std), size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(image: Image, quality: float) -> Image:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover - encoding should not fail
        return image.copy()
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _downscale_upscale(image: Image, factor: float) -> Image:
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _contrast_fade(image: Image, alpha: float) -> Image:
    mean = image.mean(axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)


def _brightness_gamma(image: Image, gamma: float) -> Image:
    lut = (np.linspace(0.0, 1.0, 256) ** float(gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(image, lut)


def _disparity_scale(image: Image, factor: float) -> Image:
    # horizontal stretch about the image centre -- changes disparity, stays matchable
    h, w = image.shape[:2]
    m = np.float32([[factor, 0, (1 - factor) * w / 2.0], [0, 1, 0]])
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _elastic_warp(image: Image, amplitude: float) -> Image:
    h, w = image.shape[:2]
    rng = np.random.default_rng(_SEED)
    smooth = max(h, w) / 20.0
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=smooth)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=smooth)
    dx = amplitude * dx / (dx.std() + 1e-6)
    dy = amplitude * dy / (dy.std() + 1e-6)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(
        image, xs + dx, ys + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _occlusion_patch(image: Image, area_fraction: float) -> Image:
    out = image.copy()
    h, w = image.shape[:2]
    side = int(round((area_fraction * h * w) ** 0.5))
    side = max(1, min(side, h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    out[y0 : y0 + side, x0 : x0 + side] = 128
    return out


def _scramble(image: Image, frac: float) -> Image:
    out = image.copy()
    h, w = image.shape[:2]
    g = 8
    bh, bw = h // g, w // g
    if bh == 0 or bw == 0:  # pragma: no cover - tiny images
        return out
    rng = np.random.default_rng(_SEED)
    cells = [(r, c) for r in range(g) for c in range(g)]
    permuted = cells.copy()
    rng.shuffle(permuted)
    src = image.copy()
    for (r, c), (pr, pc) in zip(cells, permuted, strict=True):
        out[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw] = src[
            pr * bh : (pr + 1) * bh, pc * bw : (pc + 1) * bw
        ]
    return out


register("identity", _identity, (0.0,), "anchor_low", "anchor")
register("gaussian_blur", _gaussian_blur, (0.5, 1.0, 2.0, 4.0, 8.0), "rises", "texture")
register(
    "horizontal_shift", _horizontal_shift, (1.0, 2.0, 4.0, 8.0, 16.0, 32.0), "flat", "geometric"
)
register("vertical_shift", _vertical_shift, (1.0, 2.0, 3.0, 4.0, 6.0, 8.0), "rises", "geometric")
register("gaussian_noise", _gaussian_noise, (5.0, 10.0, 20.0, 40.0, 80.0), "rises", "texture")
register("jpeg", _jpeg, (90.0, 70.0, 50.0, 30.0, 15.0, 5.0), "rises", "texture")
register("downscale_upscale", _downscale_upscale, (0.75, 0.5, 0.35, 0.25, 0.15), "rises", "texture")
register("contrast_fade", _contrast_fade, (0.8, 0.6, 0.4, 0.2, 0.1), "rises", "texture")
register("brightness_gamma", _brightness_gamma, (1.25, 1.5, 2.0, 2.5), "flat", "photometric")
register("disparity_scale", _disparity_scale, (1.02, 1.05, 1.1, 1.2), "flat", "geometric")
register("elastic_warp", _elastic_warp, (1.0, 2.0, 4.0, 8.0, 12.0), "rises", "geometric")
register("occlusion_patch", _occlusion_patch, (0.05, 0.1, 0.2, 0.35, 0.5), "rises", "structural")
register("scramble", _scramble, (1.0,), "anchor_high", "anchor")
