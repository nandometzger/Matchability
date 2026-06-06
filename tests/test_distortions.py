"""Unit tests for the distortion registry (mechanical contracts; trends live in properties)."""

import cv2
import numpy as np
import pytest

from matchability import distortions as D


def _img(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_registry_has_expected_distortions():
    names = set(D.available())
    expected = {
        "identity",
        "gaussian_blur",
        "horizontal_shift",
        "vertical_shift",
        "gaussian_noise",
        "jpeg",
        "downscale_upscale",
        "contrast_fade",
        "disparity_scale",
        "elastic_warp",
        "occlusion_patch",
        "scramble",
    }
    assert expected <= names


@pytest.mark.parametrize("name", D.available())
def test_distortion_contract_shape_dtype_determinism(name):
    img = _img()
    dist = D.get(name)
    severity = dist.severities[len(dist.severities) // 2]
    out = D.apply(name, img, severity)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # deterministic: same input + severity -> identical output
    assert np.array_equal(out, D.apply(name, img, severity))
    # must not mutate the input in place
    assert np.array_equal(img, _img())


def test_identity_is_unchanged():
    img = _img()
    out = D.apply("identity", img, D.get("identity").severities[0])
    assert np.array_equal(out, img)


def test_gaussian_blur_reduces_high_frequency():
    img = _img()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    strong = D.get("gaussian_blur").severities[-1]
    out = D.apply("gaussian_blur", img, strong)
    blurred = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    assert blurred < sharp


def test_horizontal_shift_translates_content():
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    img[:, 20] = 255  # a vertical white line at column 20
    out = D.apply("horizontal_shift", img, 5)
    col_means = out[:, :, 0].mean(axis=0)
    assert int(col_means.argmax()) == 25  # line moved 5px right


def test_occlusion_patch_covers_requested_area():
    img = _img()
    out = D.apply("occlusion_patch", img, 0.25)
    covered = np.all(out == 128, axis=2).mean()
    assert covered == pytest.approx(0.25, abs=0.05)


def test_sweep_defaults_to_coarse_severities():
    d = D.get("gaussian_blur")
    assert d.sweep() == d.severities
    assert d.sweep(fine=False) == d.severities


def test_fine_sweep_is_at_least_as_dense_as_coarse():
    for name in D.available():
        d = D.get(name)
        assert len(d.sweep(fine=True)) >= len(d.sweep(fine=False))


def test_fine_sweep_is_strictly_denser_for_continuous_distortions():
    for name in ["gaussian_blur", "horizontal_shift", "vertical_shift", "jpeg", "occlusion_patch"]:
        d = D.get(name)
        assert len(d.sweep(fine=True)) > len(d.severities)


@pytest.mark.parametrize("name", D.available())
def test_fine_sweep_endpoints_are_valid_images(name):
    img = _img()
    sweep = D.get(name).sweep(fine=True)
    for severity in (sweep[0], sweep[-1]):
        out = D.apply(name, img, severity)
        assert out.shape == img.shape and out.dtype == np.uint8
