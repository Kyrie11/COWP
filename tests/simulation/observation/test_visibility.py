import numpy as np

from ocrap.simulation.observation.visibility import grid_coords, project_occlusion_shadow


def test_dynamic_occluder_creates_unknown_shadow():
    grid = grid_coords(20.0, 1.0)
    box = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 1.5, 1.0], dtype=np.float32)
    shadow = project_occlusion_shadow(np.zeros(2), box, grid, 20.0)
    assert shadow.sum() > 0
    X, Y = grid
    assert shadow[(abs(Y) < 0.5) & (X > 8)].any()
