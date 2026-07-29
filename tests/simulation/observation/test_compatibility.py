import numpy as np

from ocrap.data.schema import Observation
from ocrap.simulation.observation.compatibility import compatibility_labels


def obs_with_box(x=None):
    boxes = np.zeros((0, 9), dtype=np.float32) if x is None else np.array([[x, 0, 0, 0, 0, 4, 2, 1.5, 1]], dtype=np.float32)
    valid = np.zeros((0,), dtype=bool) if x is None else np.ones((1,), dtype=bool)
    return Observation(np.zeros(9, dtype=np.float32), boxes, valid, np.zeros((7, 8, 8), dtype=np.float32), False, np.zeros(3, dtype=np.float32))


def test_hidden_boxes_do_not_leak_and_visible_boxes_affect_distance():
    Y, C, D = compatibility_labels([obs_with_box(None), obs_with_box(None), obs_with_box(10.0)], {"epsilon_obs": 1.0})
    assert Y[0, 1] == 1
    assert D[0, 2] > 1.0
    assert np.allclose(Y, Y.T)
    assert np.allclose(np.diag(C), 1.0)
