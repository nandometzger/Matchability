"""Unit tests for rectified epipolar-consistency filtering."""

import numpy as np

from matchability.epipolar import epipolar_consistent_rectified


def test_vertical_threshold():
    left = np.array([[100, 100], [100, 100], [100, 100]], dtype=float)
    right = np.array([[80, 101], [70, 104], [60, 98]], dtype=float)  # dy = -1, -4, +2
    mask = epipolar_consistent_rectified(left, right, tau=2.0)
    assert mask.tolist() == [True, False, True]  # |dy| = 1, 4, 2 ; tau=2 inclusive


def test_tau_boundary_is_inclusive():
    left = np.array([[0, 10]], dtype=float)
    right = np.array([[0, 12]], dtype=float)  # dy = 2
    assert epipolar_consistent_rectified(left, right, tau=2.0).tolist() == [True]
    assert epipolar_consistent_rectified(left, right, tau=1.9).tolist() == [False]


def test_disparity_bounds_filter_out_of_range():
    left = np.array([[100, 50], [100, 50], [100, 50]], dtype=float)
    right = np.array([[90, 50], [120, 50], [100, 50]], dtype=float)  # disp x_l-x_r = 10,-20,0
    mask = epipolar_consistent_rectified(left, right, tau=2.0, disparity_bounds=(0.0, 64.0))
    assert mask.tolist() == [True, False, True]  # negative disparity (-20) rejected


def test_disparity_bounds_default_off():
    left = np.array([[0, 0]], dtype=float)
    right = np.array([[999, 0]], dtype=float)  # wild disparity, but dy=0
    assert epipolar_consistent_rectified(left, right, tau=2.0).tolist() == [True]


def test_empty_input_returns_empty_mask():
    mask = epipolar_consistent_rectified(np.zeros((0, 2)), np.zeros((0, 2)), tau=2.0)
    assert mask.shape == (0,)
    assert mask.dtype == bool
