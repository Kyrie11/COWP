from __future__ import annotations

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.label.ego_candidates import _project_progress_profile_to_route
from cowp.label.natural_alternatives import _route_geometry_timed_trajectory, generate_natural_alternatives
from cowp.label.trajectory_primitives import constant_accel_trajectory
from tests.test_v16_8_16_dataset_support import _cfg, _short_route_scene


def test_empirical_retime_can_be_evidence_bounded_at_factual_endpoint() -> None:
    H = 80
    dt = 0.1
    current = np.zeros(11, dtype=np.float32)
    current[3] = 5.0
    current[5] = 5.0
    current[7] = 4.8
    current[8] = 1.9
    logged = np.zeros((20, 7), dtype=np.float32)
    logged[:, 0] = np.linspace(0.5, 10.0, 20)
    logged[:, 3] = 5.0
    logged[:, 5] = 4.8
    logged[:, 6] = 1.9

    bounded = _route_geometry_timed_trajectory(
        logged, current, H, dt, accel=0.0, nat_cfg={}, extrapolate_after_path=False
    )
    assert float(np.max(bounded[:, 0])) <= 10.0 + 1.0e-4
    assert np.all(np.isfinite(bounded))


def test_postfilter_constructive_closure_masks_singleton_basis_without_dropping_critical() -> None:
    cfg = _cfg()
    cfg["limits"]["max_natural_alternatives"] = 1
    cfg["natural"]["certificate_min_constructive_roots"] = 2
    cfg["natural"]["certificate_min_low_burden_roots"] = 2
    scene = _short_route_scene(future_valid_steps=80)
    critical = {
        "track_index": np.array([1], dtype=np.int32),
        "valid": np.array([True]),
        "mechanism_valid": np.array([True]),
        "auditability_reason": np.array([2], dtype=np.int32),
        "base_priority": np.array([PriorityRelation.EQUAL_OR_NEGOTIATED], dtype=np.int32),
        "score": np.array([5.0], dtype=np.float32),
    }
    H = cfg["time"]["future_steps"]
    fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    natural = generate_natural_alternatives(scene, critical, fallback, cfg)

    assert bool(critical["valid"][0])
    assert not bool(critical["mechanism_valid"][0])
    assert not bool(natural["valid"][0].any())
    d = natural["_diagnostics"][0]
    assert d["mechanism_valid"] is False
    assert d["auditability_finalizer"] == "insufficient_constructive_natural_basis"
    assert int(d["attempted_root_count_before_mask"]) <= 1


def test_progress_profile_can_be_projected_to_curved_lane_route() -> None:
    H = 80
    dt = 0.1
    current = np.zeros(11, dtype=np.float32)
    current[3] = 4.0
    current[5] = 4.0
    current[7] = 4.8
    current[8] = 1.9
    base = constant_accel_trajectory(current, H, dt, accel=-0.5)
    x = np.linspace(0.0, 28.0, 80, dtype=np.float32)
    y = 0.02 * x * x
    route = np.stack([x, y], axis=-1)
    projected = _project_progress_profile_to_route(base, route, current, dt)
    assert projected is not None
    assert projected.shape == base.shape
    assert np.all(np.isfinite(projected))
    assert float(np.max(projected[:, 1])) > 1.0
