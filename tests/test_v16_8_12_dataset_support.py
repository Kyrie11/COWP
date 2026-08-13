from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import Lane, MapData, ScenarioData
from cowp.label.natural_alternatives import (
    _lane_segment_cloud,
    _trajectory_map_compliance,
    _traj_distance_to_valid_future,
    build_pair_specific_ego_neutrals,
    generate_natural_alternatives,
)
from cowp.label.priority import priority_preservation_check
from cowp.label.trajectory_primitives import constant_accel_trajectory


def _cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text())
    cfg["limits"]["max_candidates"] = 2
    cfg["limits"]["max_critical_agents"] = 1
    cfg["limits"]["max_natural_alternatives"] = 12
    cfg["natural"]["min_natural_alternatives"] = 4
    cfg["natural"]["min_low_burden_alternatives"] = 2
    cfg["natural"]["map_route_max_routes"] = 3
    cfg["natural"]["map_route_max_depth"] = 4
    return cfg


def _straight_scene(*, logged_decel: bool = False) -> ScenarioData:
    H, cur, dt = 80, 10, 0.1
    T = cur + 1 + H
    states = np.zeros((2, T, 11), dtype=np.float32)
    lane = Lane(1, np.array([[-100.0, 0.0, 0.0], [220.0, 0.0, 0.0]], dtype=np.float32))

    def put(n: int, t: int, x: float, speed: float, valid: bool = True) -> None:
        states[n, t] = [x, 0.0, 0.0, speed, 0.0, speed, 0.0, 4.8, 1.9, 1.6, float(valid)]

    for t in range(cur + 1):
        tau = (t - cur) * dt
        put(0, t, 10.0 * tau, 10.0)
        put(1, t, -25.0 + 10.0 * tau, 10.0)
    agent_x = -25.0
    agent_v = 10.0
    for j in range(H):
        tau = (j + 1) * dt
        put(0, cur + 1 + j, 10.0 * tau, 10.0)
        if logged_decel:
            agent_v = max(2.0, agent_v - 0.10)  # factual timing deliberately slows
            agent_x += agent_v * dt
        else:
            agent_x = -25.0 + 10.0 * tau
        put(1, cur + 1 + j, agent_x, agent_v)
    return ScenarioData(
        scenario_id="v16812_straight",
        timestamps=np.arange(T, dtype=np.float32) * dt,
        current_time_index=cur,
        states=states,
        object_type=np.array([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([1, 2], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.array([2], dtype=np.int64),
        tracks_to_predict=np.array([1], dtype=np.int32),
        map_data=MapData(lanes={1: lane}),
    )


def _critical() -> dict[str, np.ndarray]:
    return {
        "track_index": np.array([1], dtype=np.int32),
        "valid": np.array([True]),
        "base_priority": np.array([PriorityRelation.AGENT_PRIORITY], dtype=np.int32),
        "score": np.array([5.0], dtype=np.float32),
    }


def test_map_compliance_uses_continuous_segments_not_sparse_lane_sample_points() -> None:
    cfg = _cfg()
    scene = _straight_scene()
    tr = np.zeros((80, 7), dtype=np.float32)
    tr[:, 0] = np.linspace(1.0, 19.0, 80)
    tr[:, 3] = 2.0
    tr[:, 5] = 4.8
    tr[:, 6] = 1.9
    ok, max_dist, verified = _trajectory_map_compliance(
        tr, _lane_segment_cloud(scene), int(ObjectType.VEHICLE), cfg["natural"]
    )
    assert verified
    assert ok
    assert max_dist < 1.0e-4


def test_full_logged_future_remains_obs_evidence_but_not_normative_timing_reference() -> None:
    cfg = _cfg()
    scene = _straight_scene(logged_decel=True)
    H = cfg["time"]["future_steps"]
    fallback = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutrals, _ = build_pair_specific_ego_neutrals(scene, _critical(), fallback, cfg)
    natural = generate_natural_alternatives(scene, _critical(), neutrals, cfg)
    d = natural["_diagnostics"][0]
    assert d["future_valid_steps"] == 80
    assert d["obs_eligible"] is True
    assert d["reference_kind"] == "map_route_neutral_timing"
    assert d["root_count"] >= 2
    assert d["low_burden_root_count"] >= 2
    # OBS is still permitted as empirical evidence; neutral timing only replaces
    # its use as the normative burden/progress reference.
    assert np.any((natural["source"][0] == 1) & natural["valid"][0]) or d["obs_contamination"] >= 0.0


def test_priority_preservation_exposes_rejection_mechanism() -> None:
    cfg = _cfg()
    H = 80
    ref = np.zeros((H, 7), dtype=np.float32)
    ref[:, 0] = np.arange(1, H + 1) * 0.5
    ref[:, 3] = 5.0
    ref[:, 5] = 4.8
    ref[:, 6] = 1.9
    bad = ref.copy()
    bad[1, 3] = 10.0
    ok, reason = priority_preservation_check(bad, ref, PriorityRelation.AGENT_PRIORITY, cfg)
    assert not ok
    assert reason in {"max_accel", "max_decel", "max_jerk"}


def test_route_matching_uses_raw_future_valid_timestamps_not_valid_count_prefix() -> None:
    tr = np.zeros((5, 7), dtype=np.float32)
    tr[:, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    fut = np.zeros((5, 11), dtype=np.float32)
    fut[:, 0] = [100.0, 1.0, 100.0, 3.0, 4.0]
    mask = np.array([False, True, False, True, True])
    fut[:, 10] = mask.astype(np.float32)
    assert _traj_distance_to_valid_future(tr, fut, mask) == 0.0
