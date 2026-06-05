"""Contract tests for the classical (SIFT) matcher backend."""

import numpy as np

from matchability.matchers.classical import ClassicalMatcher
from matchability.types import LeftToRightMatches


def test_detect_returns_xy_array(textured_image):
    m = ClassicalMatcher()
    kp = m.detect(textured_image, n=200)
    assert kp.ndim == 2 and kp.shape[1] == 2
    assert 0 < len(kp) <= 200


def test_match_is_aligned_to_reference_keypoints(textured_image):
    m = ClassicalMatcher()
    kp = m.detect(textured_image, n=200)
    res = m.match(textured_image, kp, textured_image)
    assert isinstance(res, LeftToRightMatches)
    assert len(res) == len(kp)


def test_match_self_maps_each_keypoint_to_itself(textured_image):
    m = ClassicalMatcher()
    kp = m.detect(textured_image, n=200)
    res = m.match(textured_image, kp, textured_image)
    valid = res.valid
    assert valid.sum() > 0.5 * len(kp)  # most keypoints find their twin
    # self-match: right coordinate ~= left coordinate (sub-pixel)
    dxy = np.linalg.norm(kp[valid] - res.right_xy[valid], axis=1)
    assert np.median(dxy) < 1.0


def test_match_handles_empty_reference():
    m = ClassicalMatcher()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    res = m.match(img, np.zeros((0, 2)), img)
    assert len(res) == 0
