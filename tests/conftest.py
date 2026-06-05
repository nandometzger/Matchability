"""Shared fixtures: deterministic, SIFT-friendly synthetic textures and stereo pairs."""

import cv2
import numpy as np
import pytest


def make_textured_image(height: int = 256, width: int = 256, n_shapes: int = 320, seed: int = 0):
    """A deterministic richly-textured RGB image (blobs/edges/corners for keypoints)."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 127, dtype=np.uint8)
    for _ in range(n_shapes):
        kind = int(rng.integers(0, 3))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if kind == 0:
            center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            radius = int(rng.integers(2, max(3, min(height, width) // 12)))
            cv2.circle(img, center, radius, color, -1)
        elif kind == 1:
            p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            p2 = (
                p1[0] + int(rng.integers(3, width // 8)),
                p1[1] + int(rng.integers(3, height // 8)),
            )
            cv2.rectangle(img, p1, p2, color, -1)
        else:
            p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            p2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            cv2.line(img, p1, p2, color, int(rng.integers(1, 3)))
    return img


def make_stereo_pair(size: int = 256, disparity: int = 16, seed: int = 0):
    """A rectified synthetic stereo pair (left, right_gt): pure horizontal disparity, dy=0."""
    scene = make_textured_image(height=size, width=size + disparity, seed=seed)
    left = np.ascontiguousarray(scene[:, 0:size])
    right = np.ascontiguousarray(scene[:, disparity : disparity + size])
    return left, right


@pytest.fixture
def textured_image():
    return make_textured_image(seed=1)


@pytest.fixture
def stereo_pair():
    return make_stereo_pair(seed=2)
