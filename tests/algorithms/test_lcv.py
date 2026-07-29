import math

import numpy as np

from ocrap.algorithms.lcv import weighted_lcvar


def test_weighted_lcvar_lower_tail_boundary_mass():
    scores = np.array([-2.0, 0.0, 10.0])
    weights = np.array([0.1, 0.4, 0.5])
    assert math.isclose(weighted_lcvar(scores, weights, 0.2), -1.0, rel_tol=1e-6)


def test_weighted_lcvar_zero_weights_fallback():
    scores = np.array([1.0, 3.0])
    assert math.isclose(weighted_lcvar(scores, np.zeros(2), 0.5), 1.0, rel_tol=1e-6)
