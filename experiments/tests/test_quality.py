"""Unit tests for SSIM / PSNR helpers."""

import numpy as np
import pytest

from experiments.lib.quality import psnr, ssim
from matchability import distortions as D


@pytest.fixture
def textured_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_ssim_identical_is_one(textured_image):
    assert ssim(textured_image, textured_image) == 1.0


def test_psnr_identical_is_inf(textured_image):
    assert psnr(textured_image, textured_image) == float("inf")


def test_blur_lowers_ssim_and_psnr(textured_image):
    blurred = D.apply("gaussian_blur", textured_image, 8.0)
    assert ssim(textured_image, blurred) < 1.0
    assert np.isfinite(psnr(textured_image, blurred))


def test_heavier_blur_lowers_ssim_further(textured_image):
    light = D.apply("gaussian_blur", textured_image, 1.0)
    heavy = D.apply("gaussian_blur", textured_image, 8.0)
    assert ssim(textured_image, heavy) < ssim(textured_image, light)
