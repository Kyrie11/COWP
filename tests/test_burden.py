from __future__ import annotations

import numpy as np

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.label.burden import adaptive_beta, compute_burden
from cowp.label.trajectory_primitives import constant_accel_trajectory


def test_mild_vs_hard_braking_burden(cfg):
    current = np.array([0, 0, 0, 10, 0, 10, 0, 4.8, 1.9, 1.6, 1], dtype=np.float32)
    mild = constant_accel_trajectory(current, 80, 0.1, accel=-1.0)
    hard = constant_accel_trajectory(current, 80, 0.1, accel=-6.0)
    beta = adaptive_beta(None, ObjectType.VEHICLE, PriorityRelation.EQUAL_OR_NEGOTIATED, cfg)
    b_mild, _ = compute_burden(mild, None, cfg, ObjectType.VEHICLE, natural_ref=constant_accel_trajectory(current, 80, 0.1, accel=0.0))
    b_hard, comps_hard = compute_burden(hard, None, cfg, ObjectType.VEHICLE, natural_ref=constant_accel_trajectory(current, 80, 0.1, accel=0.0))
    assert b_mild < beta
    assert b_hard > b_mild
    assert comps_hard[0] >= 0.9


def test_option_collapse_component(cfg):
    current = np.array([0, 0, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1], dtype=np.float32)
    tr = constant_accel_trajectory(current, 80, 0.1, accel=0.0)
    _, comps = compute_burden(tr, None, cfg, ObjectType.VEHICLE, natural_ref=tr, option_loss=0.8)
    assert np.isclose(float(comps[4]), 0.8, atol=1e-6)
