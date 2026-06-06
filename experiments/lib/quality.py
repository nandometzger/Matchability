"""Reference image-quality metrics (SSIM, PSNR) for the sensitivity overlays."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def _gray(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.float64)


def psnr(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` for identical images)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def ssim(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    """Mean structural similarity (Wang et al. 2004) on the luma channel."""
    ga, gb = _gray(a), _gray(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ksize, sigma = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(ga, ksize, sigma)
    mu_b = cv2.GaussianBlur(gb, ksize, sigma)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    var_a = cv2.GaussianBlur(ga * ga, ksize, sigma) - mu_a2
    var_b = cv2.GaussianBlur(gb * gb, ksize, sigma) - mu_b2
    cov_ab = cv2.GaussianBlur(ga * gb, ksize, sigma) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map.mean())
