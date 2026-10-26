from __future__ import annotations

import numpy as np

from cowp.geometry.boxes import obb_overlap, obb_distance
from cowp.geometry.ttc import pairwise_ttc
from cowp.geometry.rss_like import longitudinal_safety_distance


def test_obb_overlap_parallel_and_separated():
    a = np.array([0, 0, 0, 0, 0, 4, 2], dtype=np.float32)
    b = np.array([1, 0, 0, 0, 0, 4, 2], dtype=np.float32)
    c = np.array([10, 0, 0, 0, 0, 4, 2], dtype=np.float32)
    assert obb_overlap(a, b)
    assert not obb_overlap(a, c)
    assert obb_distance(a, c) > 0


def test_obb_overlap_orthogonal():
    a = np.array([0, 0, 0, 0, 0, 4, 2], dtype=np.float32)
    b = np.array([0, 0, np.pi/2, 0, 0, 4, 2], dtype=np.float32)
    assert obb_overlap(a, b)


def test_ttc_closing_and_opening():
    ttc = pairwise_ttc(np.array([[0., 0.]]), np.array([[5., 0.]]), np.array([[10., 0.]]), np.array([[0., 0.]]))
    assert 1.9 < ttc[0] < 2.1
    opening = pairwise_ttc(np.array([[0., 0.]]), np.array([[0., 0.]]), np.array([[10., 0.]]), np.array([[5., 0.]]))
    assert np.isinf(opening[0])


def test_rss_distance_positive():
    assert longitudinal_safety_distance(10.0, 5.0) > 2.0
