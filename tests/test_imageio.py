"""Unit tests for image coercion + resizing."""

import numpy as np
import pytest

from matchability.imageio_util import load_image, resize_max_side


def test_uint8_rgb_passthrough():
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    a[1, 2] = [10, 20, 30]
    out = load_image(a)
    assert out.shape == (4, 6, 3)
    assert out.dtype == np.uint8
    assert tuple(out[1, 2]) == (10, 20, 30)


def test_grayscale_becomes_rgb():
    g = np.full((4, 5), 100, dtype=np.uint8)
    out = load_image(g)
    assert out.shape == (4, 5, 3)
    assert np.all(out == 100)


def test_rgba_alpha_is_dropped():
    a = np.zeros((3, 3, 4), dtype=np.uint8)
    a[..., :3] = 50
    out = load_image(a)
    assert out.shape == (3, 3, 3)
    assert np.all(out == 50)


def test_float_unit_range_scales_to_uint8():
    f = np.full((2, 2, 3), 0.5, dtype=np.float32)
    out = load_image(f)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] in (127, 128)


def test_pil_image():
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (8, 6), (10, 20, 30))
    out = load_image(img)
    assert out.shape == (6, 8, 3)
    assert tuple(out[0, 0]) == (10, 20, 30)


def test_path(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    arr = np.zeros((6, 8, 3), dtype=np.uint8)
    arr[..., 0] = 200  # red
    path = tmp_path / "x.png"
    pil.fromarray(arr).save(path)
    out = load_image(str(path))
    assert out.shape == (6, 8, 3)
    assert tuple(out[0, 0]) == (200, 0, 0)


def test_torch_chw_float_tensor():
    torch = pytest.importorskip("torch")
    t = torch.zeros(3, 4, 5)
    t[0] = 1.0  # red channel
    out = load_image(t)
    assert out.shape == (4, 5, 3)
    assert tuple(out[0, 0]) == (255, 0, 0)


def test_resize_max_side_downscales():
    a = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_max_side(a, 50).shape == (25, 50, 3)


def test_resize_does_not_upscale():
    a = np.zeros((10, 20, 3), dtype=np.uint8)
    assert resize_max_side(a, 100).shape == (10, 20, 3)


def test_load_image_with_max_side():
    a = np.zeros((100, 200, 3), dtype=np.uint8)
    assert load_image(a, max_side=50).shape == (25, 50, 3)
