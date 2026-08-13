from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from cowp.core.constants import ObjectType, PriorityRelation
from cowp.core.types import Lane, MapData, ScenarioData
from cowp.label.audit_relevance import compute_candidate_agent_audit
from cowp.label.burden import compute_burden
from cowp.label.natural_alternatives import (
    build_pair_specific_ego_neutrals,
    generate_natural_alternatives,
)
from cowp.label.safe_responses import generate_safe_responses, root_conditioned_recovery_search
from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.label.witness import certify_witnesses


def _cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text())
    cfg["limits"]["max_candidates"] = 2
    cfg["limits"]["max_critical_agents"] = 1
    cfg["limits"]["max_natural_alternatives"] = 12
    cfg["limits"]["max_safe_responses"] = 12
    cfg["natural"]["min_natural_alternatives"] = 4
    cfg["natural"]["min_low_burden_alternatives"] = 2
    cfg["natural"]["map_route_max_routes"] = 3
    cfg["natural"]["map_route_max_depth"] = 4
    cfg["response"]["max_response_primitives_per_agent"] = 24
    cfg["response"]["safe_budget_search"]["max_return"] = 6
    return cfg


def _rear_follow_scene(*, short_agent_future: bool = False, curved: bool = False) -> ScenarioData:
    H = 80
    cur = 10
    T = cur + 1 + H
    dt = 0.1
    states = np.zeros((2, T, 11), dtype=np.float32)
    if curved:
        # Quarter-circle-like lane with enough downstream arc length.
        theta = np.linspace(0.0, 1.45, 180, dtype=np.float32)
        lane_xy = np.stack([60.0 * np.sin(theta), 60.0 * (1.0 - np.cos(theta)), np.zeros_like(theta)], axis=-1)
    else:
        lane_xy = np.stack([
            np.linspace(-80.0, 220.0, 181, dtype=np.float32),
            np.zeros(181, dtype=np.float32),
            np.zeros(181, dtype=np.float32),
        ], axis=-1)
    lane = Lane(1, lane_xy)

    # Current states are deliberately aligned with the first usable lane segment.
    if curved:
        ego_pos = np.array([0.0, 0.0], dtype=np.float32)
        ego_yaw = 0.0
        agent_pos = np.array([-30.0, 0.0], dtype=np.float32)
        # Add an upstream straight entry lane so the rear actor projects correctly.
        entry_xy = np.array([[-80.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        entry = Lane(0, entry_xy, exit_lanes=(1,))
        lane = Lane(1, lane_xy, entry_lanes=(0,))
        map_data = MapData(lanes={0: entry, 1: lane})
    else:
        ego_pos = np.array([0.0, 0.0], dtype=np.float32)
        ego_yaw = 0.0
        agent_pos = np.array([-30.0, 0.0], dtype=np.float32)
        map_data = MapData(lanes={1: lane})

    def fill_state(n: int, t: int, xy: np.ndarray, speed: float, yaw: float, valid: bool = True):
        states[n, t] = [xy[0], xy[1], 0.0, speed*np.cos(yaw), speed*np.sin(yaw), speed, yaw, 4.8, 1.9, 1.6, float(valid)]

    # History/current: ego ahead, rear actor closing slowly. The pair-specific
    # neutral should prefer keep/mild acceleration over a global ego brake.
    for t in range(cur + 1):
        tau = (t - cur) * dt
        fill_state(0, t, ego_pos + np.array([10.0*tau, 0.0], dtype=np.float32), 10.0, ego_yaw)
        fill_state(1, t, agent_pos + np.array([10.0*tau, 0.0], dtype=np.float32), 10.0, ego_yaw)

    # Future geometry is valid for ego. Agent can be deliberately truncated to
    # model WOMD valid=0 rows and exercise lane-graph reconstruction.
    for j in range(H):
        tau = (j + 1) * dt
        fill_state(0, cur + 1 + j, ego_pos + np.array([10.0*tau, 0.0], dtype=np.float32), 10.0, 0.0)
        av = (j < 12) if short_agent_future else True
        fill_state(1, cur + 1 + j, agent_pos + np.array([10.0*tau, 0.0], dtype=np.float32), 10.0, 0.0, valid=av)
        if not av:
            states[1, cur + 1 + j, :10] = 0.0

    return ScenarioData(
        scenario_id="rear_follow",
        timestamps=np.arange(T, dtype=np.float32) * dt,
        current_time_index=cur,
        states=states,
        object_type=np.array([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.array([100, 200], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.array([200], dtype=np.int64),
        tracks_to_predict=np.array([1], dtype=np.int32),
        map_data=map_data,
    )


def _critical() -> dict[str, np.ndarray]:
    return {
        "track_index": np.array([1], dtype=np.int32),
        "valid": np.array([True]),
        "base_priority": np.array([PriorityRelation.AGENT_PRIORITY], dtype=np.int32),
        "score": np.array([5.0], dtype=np.float32),
    }


def test_pair_specific_neutral_does_not_force_rear_follower_to_absorb_global_brake():
    cfg = _cfg()
    scene = _rear_follow_scene()
    H = cfg["time"]["future_steps"]
    ego_cur = scene.ego_current
    global_brake = constant_accel_trajectory(ego_cur, H, 0.1, accel=-2.0)
    neutrals, diag = build_pair_specific_ego_neutrals(scene, _critical(), global_brake, cfg)

    assert neutrals.shape == (1, H, 7)
    assert diag[0]["valid"]
    assert not diag[0]["neutral_actor_unsafe"]
    # Regression against the old one-global-YIELD neutral. In a rear-follow pair,
    # the chosen pressure-removing intervention should not simply be that brake.
    assert float(np.mean(np.linalg.norm(neutrals[0, :, :2] - global_brake[:, :2], axis=-1))) > 0.25


def test_short_invalid_womd_future_uses_lane_route_and_keeps_two_low_burden_roots():
    cfg = _cfg()
    scene = _rear_follow_scene(short_agent_future=True)
    H = cfg["time"]["future_steps"]
    global_brake = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutrals, _ = build_pair_specific_ego_neutrals(scene, _critical(), global_brake, cfg)
    natural = generate_natural_alternatives(scene, _critical(), neutrals, cfg)
    d = natural["_diagnostics"][0]

    assert d["future_valid_steps"] == 12
    assert d["obs_eligible"] is False
    assert d["reference_kind"] == "map_route_neutral_timing"
    assert d["root_count"] >= 2
    assert d["low_burden_root_count"] >= 2
    assert int(np.sum(natural["valid"][0])) >= 2
    assert int(np.sum(natural["map_verified"][0] & natural["valid"][0])) >= 2


def test_audit_identity_cache_is_semantically_exact_for_response_and_root_oracle():
    """The speed path must reuse computation, not change labels."""
    cfg = _cfg()
    scene = _rear_follow_scene(short_agent_future=True)
    H = cfg["time"]["future_steps"]
    global_brake = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    neutrals, _ = build_pair_specific_ego_neutrals(scene, _critical(), global_brake, cfg)
    natural = generate_natural_alternatives(scene, _critical(), neutrals, cfg)

    candidates = {
        "trajectory": np.zeros((2, H, 7), dtype=np.float32),
        "valid": np.array([True, True]),
    }
    candidates["trajectory"][0] = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=0.0)
    candidates["trajectory"][1] = constant_accel_trajectory(scene.ego_current, H, 0.1, accel=-2.0)
    audit = compute_candidate_agent_audit(scene, candidates, _critical(), natural, cfg)

    # Only compare the response stage where the fresh private cache is optional.
    response_fast = generate_safe_responses(scene, candidates, _critical(), natural, cfg, audit=audit)
    audit_slow = {k: v for k, v in audit.items() if not str(k).startswith("_root_direct_")}
    response_slow = generate_safe_responses(scene, candidates, _critical(), natural, cfg, audit=audit_slow)
    for key in response_fast:
        np.testing.assert_array_equal(response_fast[key], response_slow[key])

    m = int(np.where(natural["valid"][0])[0][0])
    root = natural["traj"][0, m]
    beta = float(natural["beta"][0])
    base = root_conditioned_recovery_search(
        root, candidates["trajectory"][0], cfg, object_type=int(ObjectType.VEHICLE),
        beta=beta, rho=PriorityRelation.AGENT_PRIORITY,
    )
    cached = root_conditioned_recovery_search(
        root, candidates["trajectory"][0], cfg, object_type=int(ObjectType.VEHICLE),
        beta=beta, rho=PriorityRelation.AGENT_PRIORITY,
        identity_cache=(
            float(audit["_root_direct_burden_exact"][0, 0, m]),
            not bool(audit["root_unsafe"][0, 0, m]),
        ),
    )
    assert base == cached
