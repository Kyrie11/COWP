from __future__ import annotations

from pathlib import Path
import importlib
import numpy as np
import yaml

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import Lane, MapData, ScenarioData
from cowp.label.natural_alternatives import (
    _driveway_polygons,
    _lane_segment_cloud,
    _trajectory_map_compliance,
    build_pair_specific_ego_neutrals,
    generate_natural_alternatives,
)
from cowp.label.trajectory_primitives import constant_accel_trajectory, resample_logged
_screen = importlib.import_module("cowp.scripts.58_screen_v16_8_9_causal_audit_probe")
_smoke_max_metric_pass = _screen._smoke_max_metric_pass
_smoke_min_metric_pass = _screen._smoke_min_metric_pass
_wilson_interval = _screen._wilson_interval


def _cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text())
    cfg["limits"]["max_candidates"] = 2
    cfg["limits"]["max_critical_agents"] = 1
    cfg["limits"]["max_natural_alternatives"] = 12
    return cfg


def test_driveway_polygon_is_hd_map_evidence_without_lane_centerline() -> None:
    cfg = _cfg()
    poly = np.array([[0.0, -3.0, 0.0], [30.0, -3.0, 0.0], [30.0, 3.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    scene = ScenarioData(
        scenario_id="driveway",
        timestamps=np.arange(91, dtype=np.float32) * 0.1,
        current_time_index=10,
        states=np.zeros((1, 91, 11), dtype=np.float32),
        object_type=np.array([ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([1], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.zeros(0, dtype=np.int64),
        tracks_to_predict=np.zeros(0, dtype=np.int32),
        map_data=MapData(driveways={7: poly}),
    )
    tr = np.zeros((80, 7), dtype=np.float32)
    tr[:, 0] = np.linspace(0.5, 29.5, 80)
    tr[:, 5:7] = (4.8, 1.9)
    ok, max_dist, verified = _trajectory_map_compliance(
        tr, _lane_segment_cloud(scene), int(ObjectType.VEHICLE), cfg["natural"],
        driveway_polygons=_driveway_polygons(scene),
    )
    assert verified
    assert ok
    assert max_dist == 0.0


def test_identity_observational_root_preserves_logged_positions_even_when_velocity_disagrees() -> None:
    H = 80
    logged = np.zeros((H, 7), dtype=np.float32)
    t = np.linspace(0.0, 1.0, H)
    logged[:, 0] = 20.0 * t
    logged[:, 1] = 3.0 * np.sin(t * np.pi / 2.0)
    logged[:, 2] = 0.0
    # Deliberately inconsistent stored velocity; old resampling integrated this
    # and drifted away from the factual WOMD positions.
    logged[:, 3] = 12.0
    logged[:, 4] = 0.0
    logged[:, 5:7] = (4.8, 1.9)
    current = np.zeros(11, dtype=np.float32)
    current[7:9] = (4.8, 1.9)
    out = resample_logged(logged, H, 0, 1.0, 0.0, current=current, dt=0.1)
    np.testing.assert_allclose(out[:, :2], logged[:, :2], atol=1e-7)


def _short_route_scene(future_valid_steps: int = 35) -> ScenarioData:
    H, cur, dt = 80, 10, 0.1
    T = cur + 1 + H
    states = np.zeros((2, T, 11), dtype=np.float32)
    for t in range(cur + 1):
        states[0, t] = [0.5 * (t-cur), 0, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1]
        states[1, t] = [0.5 * (t-cur), 5, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1]
    for j in range(H):
        states[0, cur+1+j] = [0.5*(j+1), 0, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1]
        valid = j < int(future_valid_steps)
        states[1, cur+1+j] = [0.5*(j+1), 5, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, float(valid)]
    # Actor starts on a valid lane but only ~15 m remain, much shorter than the
    # full-horizon route required at 5 m/s.  The lane polyline exists, yet no
    # full 8 s retiming should be possible.
    lane = Lane(7, np.array([[0.0, 5.0, 0.0], [15.0, 5.0, 0.0]], dtype=np.float32))
    return ScenarioData(
        scenario_id="short_route",
        timestamps=np.arange(T, dtype=np.float32)*dt,
        current_time_index=cur,
        states=states,
        object_type=np.array([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([1,2], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.array([2], dtype=np.int64),
        tracks_to_predict=np.array([1], dtype=np.int32),
        map_data=MapData(lanes={7: lane}),
    )


def test_nonempty_but_too_short_lane_does_not_falsely_authorize_mechanism_supervision() -> None:
    cfg = _cfg()
    scene = _short_route_scene()
    critical = {
        "track_index": np.array([1], dtype=np.int32),
        "valid": np.array([True]),
        "mechanism_valid": np.array([True]),
        "auditability_reason": np.array([0], dtype=np.int32),
        "base_priority": np.array([PriorityRelation.EQUAL_OR_NEGOTIATED], dtype=np.int32),
        "score": np.array([5.0], dtype=np.float32),
    }
    H = cfg["time"]["future_steps"]
    fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutral, _ = build_pair_specific_ego_neutrals(scene, critical, fallback, cfg)
    natural = generate_natural_alternatives(scene, critical, neutral, cfg)
    d = natural["_diagnostics"][0]
    assert d["mechanism_valid"] is False
    assert not bool(critical["mechanism_valid"][0])
    assert not bool(natural["valid"][0].any())


def test_short_map_polyline_does_not_block_empirical_fallback_when_full_future_exists() -> None:
    cfg = _cfg()
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
    neutral, _ = build_pair_specific_ego_neutrals(scene, critical, fallback, cfg)
    natural = generate_natural_alternatives(scene, critical, neutral, cfg)
    d = natural["_diagnostics"][0]
    assert d["route_polyline_count"] >= 1
    assert d["full_horizon_map_route_count"] == 0
    assert d["empirical_corridor_eligible"] is True
    assert d["root_count"] >= 6
    assert d["low_burden_root_count"] >= 2


def test_smoke_pbtr_borderline_is_inconclusive_not_a_gross_failure() -> None:
    # v16.8.15 smoke PBTR: 23/45 = .5111, point estimate is above .50,
    # but the 95% Wilson lower bound is ~.37.  Smoke should allow a strict probe
    # if all semantic/data-support gates pass; strict still uses the point limit.
    ci = _wilson_interval(23, 45)
    assert ci["rate"] > 0.50
    assert _smoke_max_metric_pass(ci, 0.50)
    # A truly bad smoke must still fail when the entire interval exceeds .50.
    bad = _wilson_interval(44, 45)
    assert not _smoke_max_metric_pass(bad, 0.50)
    # Symmetric minimum-metric behavior.
    assert _smoke_min_metric_pass(_wilson_interval(15, 48), 0.30)
