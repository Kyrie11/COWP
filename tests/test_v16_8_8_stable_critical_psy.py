from __future__ import annotations

import numpy as np

from cowp.core.constants import ProposalSource
from cowp.label.critical_agents import select_critical_agents
from cowp.label.ego_candidates import generate_ego_candidates
from cowp.label.trajectory_primitives import constant_accel_trajectory, smooth_terminal_speed_arrival_trajectory


def test_fixed_anchor_critical_selection_is_proposal_bank_independent(toy_scene, cfg) -> None:
    cfg = dict(cfg)
    cfg["critical"] = dict(cfg.get("critical", {}))
    cfg["critical"]["selection_reference_mode"] = "fixed_anchor_v1"
    H = int(cfg["time"]["future_steps"])
    dt = float(cfg["time"]["dt"])
    cur = toy_scene.states[toy_scene.sdc_track_index, toy_scene.current_time_index]
    keep = constant_accel_trajectory(cur, H, dt, accel=0.0)
    aggressive = constant_accel_trajectory(cur, H, dt, accel=2.0, lateral_offset=3.5)
    bank_a = {"trajectory": np.stack([keep]), "valid": np.asarray([True])}
    bank_b = {"trajectory": np.stack([keep, aggressive]), "valid": np.asarray([True, True])}
    a = select_critical_agents(toy_scene, cfg, bank_a)
    b = select_critical_agents(toy_scene, cfg, bank_b)
    assert np.array_equal(a["track_index"], b["track_index"])
    assert np.array_equal(a["valid"], b["valid"])
    assert np.allclose(a["score"], b["score"])


def test_priority_smooth_yield_quintic_hits_low_speed_arrival() -> None:
    cur = np.asarray([0, 0, 0, 8, 0, 8, 0, 4.8, 1.9, 1.6, 1], dtype=np.float32)
    tr = smooth_terminal_speed_arrival_trajectory(
        cur, 80, 0.1, distance_m=28.0, target_time_s=5.0, terminal_speed_mps=1.5, initial_accel_mps2=-2.0
    )
    assert tr is not None
    # Around the target time, position and speed match the requested smooth yield.
    assert abs(float(tr[49, 0]) - 28.0) < 0.35
    assert abs(float(np.linalg.norm(tr[49, 3:5])) - 1.5) < 0.35
    assert float(np.linalg.norm(tr[9, 3:5])) < 8.0 - 0.5


def test_pchr_is_disabled_and_psy_provenance_available_in_main_config(toy_scene) -> None:
    from cowp.core.config import load_config
    cfg = load_config("configs/label_cowp_v16_8.yaml")
    # Shrink expensive downstream-independent dimensions only; candidate generator semantics remain main-config.
    cfg["time"]["future_steps"] = 80
    cand = generate_ego_candidates(toy_scene, cfg)
    src = np.asarray(cand["proposal_source"], dtype=np.int64)
    valid = np.asarray(cand["valid"], dtype=bool)
    assert not np.any(valid & (src == int(ProposalSource.PRIORITY_HOLD_RELEASE)))
    # PSY can be scene-conditional, so provenance enum must at least be accepted and diagnostics recorded.
    dbg = cand.get("_proposal_debug", {})
    assert "accepted_by_source" in dbg
    assert "attempted_by_source" in dbg


def test_psy_is_generated_for_protected_stop_control_interaction() -> None:
    from cowp.core.config import load_config
    from cowp.core.constants import ObjectType
    from cowp.core.types import Lane, MapData, ScenarioData
    from cowp.geometry.lane_graph import ConflictRegion

    def state(x: float, y: float, vx: float, vy: float, yaw: float) -> np.ndarray:
        s = np.zeros(11, dtype=np.float32)
        s[:] = [x, y, 0.0, vx, vy, float(np.hypot(vx, vy)), yaw, 4.8, 1.9, 1.6, 1.0]
        return s

    cfg = load_config("configs/label_cowp_v16_8.yaml")
    cfg["candidate"]["map_filter_enabled"] = False
    cfg["limits"]["max_candidates"] = 96
    steps, cur = 91, 10
    states = np.zeros((2, steps, 11), dtype=np.float32)
    for t in range(steps):
        tau = (t - cur) * 0.1
        states[0, t] = state(10.0 * tau, 0.0, 10.0, 0.0, 0.0)
        states[1, t] = state(30.0, -30.0 + 5.0 * tau, 0.0, 5.0, np.pi / 2.0)
    map_data = MapData(lanes={
        1: Lane(1, np.asarray([[-20.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float32), controlled_by_stop=True),
        2: Lane(2, np.asarray([[30.0, -60.0, 0.0], [30.0, 60.0, 0.0]], dtype=np.float32), controlled_by_stop=False),
    })
    scene = ScenarioData(
        scenario_id="psy-protected-toy",
        timestamps=np.arange(steps, dtype=np.float32) * 0.1,
        current_time_index=cur,
        states=states,
        object_type=np.asarray([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.asarray([0, 1], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.asarray([1], dtype=np.int64),
        tracks_to_predict=np.asarray([1], dtype=np.int32),
        map_data=map_data,
    )
    region = ConflictRegion(3, "LANE_INTERSECTION", np.asarray([30.0, 0.0], dtype=np.float32), 3.0, (1, 2))
    out = generate_ego_candidates(scene, cfg, conflict_regions=[region])
    valid = np.asarray(out["valid"], dtype=bool)
    psy = valid & (np.asarray(out["proposal_source"], dtype=np.int64) == int(ProposalSource.PRIORITY_SMOOTH_YIELD))
    assert psy.any()
    assert np.all(np.asarray(out["proposal_timing_side"])[psy] == 1)
    assert np.all(np.asarray(out["proposal_target_agent_index"])[psy] == 1)
    assert int(out["_proposal_debug"]["accepted_by_source"].get("PRIORITY_SMOOTH_YIELD", 0)) > 0
