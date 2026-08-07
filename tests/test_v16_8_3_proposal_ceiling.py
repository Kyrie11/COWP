from __future__ import annotations

import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import ObjectType, ProposalSource
from cowp.core.types import MapData, ScenarioData
from cowp.geometry.lane_graph import ConflictRegion
from cowp.label.ego_candidates import _candidate_valid, generate_ego_candidates
from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.waymax_eval.rollout import _LearnedMetricsAccumulator


def _state(x: float, y: float, vx: float, vy: float, heading: float) -> np.ndarray:
    s = np.zeros(11, dtype=np.float32)
    s[:] = [x, y, 0.0, vx, vy, float(np.hypot(vx, vy)), heading, 4.8, 1.9, 1.6, 1.0]
    return s


def test_offline_candidate_jerk_filter_uses_configured_prefix_and_percentile() -> None:
    cfg = load_config("configs/label_cowp_v16_8.yaml")
    current = _state(0.0, 0.0, 10.0, 0.0, 0.0)
    traj = constant_accel_trajectory(current, 80, 0.1, accel=2.5)
    assert _candidate_valid(traj, cfg)

    strict = load_config("configs/label_cowp_v16_8.yaml")
    strict["candidate"]["ignore_initial_jerk_steps"] = 0
    strict["candidate"]["jerk_check_percentile"] = 100.0
    assert not _candidate_valid(traj, strict)


def test_rmr_bcte_generates_bidirectional_proposals_with_provenance() -> None:
    cfg = load_config("configs/label_cowp_v16_8.yaml")
    cfg["candidate"]["map_filter_enabled"] = False
    cfg["limits"]["max_candidates"] = 64
    cfg["candidate"]["timing_envelope_gap_s"] = [0.8, 1.4]
    cfg["candidate"]["timing_envelope_max_regions"] = 2
    cfg["candidate"]["timing_envelope_max_candidates"] = 12

    steps = 91
    cur = 10
    states = np.zeros((2, steps, 11), dtype=np.float32)
    for t in range(steps):
        tau = (t - cur) * 0.1
        states[0, t] = _state(10.0 * tau, 0.0, 10.0, 0.0, 0.0)
        states[1, t] = _state(30.0, -30.0 + 5.0 * tau, 0.0, 5.0, np.pi / 2.0)
    scene = ScenarioData(
        scenario_id="rmr-bcte-toy",
        timestamps=np.arange(steps, dtype=np.float32) * 0.1,
        current_time_index=cur,
        states=states,
        object_type=np.asarray([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.asarray([0, 1], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.asarray([1], dtype=np.int64),
        tracks_to_predict=np.asarray([1], dtype=np.int32),
        map_data=MapData(),
    )
    regions = [
        ConflictRegion(3, "LANE_INTERSECTION", np.asarray([30.0, 0.0], dtype=np.float32), 3.0, (1, 2)),
        ConflictRegion(4, "LANE_INTERSECTION", np.asarray([50.0, 0.0], dtype=np.float32), 3.0, (1, 3)),
    ]
    out = generate_ego_candidates(scene, cfg, conflict_regions=regions)
    valid = out["valid"]
    rmr = valid & (out["proposal_source"] == int(ProposalSource.ROBUST_BCTE))
    assert rmr.any()
    assert set(out["proposal_timing_side"][rmr].tolist()) == {-1, 1}
    assert np.all(out["proposal_target_agent_index"][rmr] == 1)
    assert np.all(np.isfinite(out["proposal_target_time_s"][rmr]))
    assert np.all(np.isfinite(out["proposal_accel_mps2"][rmr]))
    assert np.all(out["proposal_entry_distance_m"][rmr] > 0.0)
    assert np.all(out["proposal_target_tta_error_s"][rmr] >= 0.0)
    assert np.max(out["proposal_target_tta_error_s"][rmr]) <= cfg["candidate"]["timing_envelope_max_target_tta_error_s"] + 1e-6


def test_proposal_floor_metrics_separate_bank_ceiling_from_selector_error() -> None:
    acc = _LearnedMetricsAccumulator()
    traj = np.zeros((2, 3, 7), dtype=np.float32)
    traj[:, :, 5:7] = np.asarray([4.8, 1.9], dtype=np.float32)
    traj[0, :, 0] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    traj[1, :, 0] = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    label_hard = {
        "cowp/candidates/trajectory": traj,
        "cowp/candidates/valid": np.asarray([True, True]),
        "cowp/candidates/conventional_safe": np.asarray([True, False]),
        "cowp/candidates/noncoercive_feasible": np.asarray([False, False]),
        "cowp/candidates/false_safe": np.asarray([True, False]),
        "cowp/critical/valid": np.zeros(0, dtype=bool),
    }
    label_recoverable = {
        "cowp/candidates/trajectory": traj,
        "cowp/candidates/valid": np.asarray([True, True]),
        "cowp/candidates/conventional_safe": np.asarray([True, True]),
        "cowp/candidates/noncoercive_feasible": np.asarray([True, False]),
        "cowp/candidates/false_safe": np.asarray([False, True]),
        "cowp/critical/valid": np.zeros(0, dtype=bool),
    }
    acc.add_selection(0, np.asarray([True, False]), label_hard)
    acc.add_selection(0, np.asarray([True, False]), label_recoverable, fallback_used=True)
    metrics = acc.finish(auprc=1.0, rank_good=1, rank_total=1, witness_threshold=0.5)
    assert metrics["ProposalCoverage/BestCaseSelectedFalseSafeLowerBound"] == 0.5
    assert metrics["Selector/NCFSelectionRecallGivenAvailable"] == 1.0
    assert metrics["Selector/FalseSafeExcessAboveProposalFloor"] == 0.0
    assert metrics["FallbackSelection/AnyNCFCandidateSceneRate"] == 1.0
    assert metrics["ProposalDiagnostics/Version"] == "v16_8_4_boundary_consistent_proposal_floor"


def test_rmr_bcte_pass_after_is_boundary_time_consistent() -> None:
    """Regression: v16.8.3 tagged a 5.6s pass-after that entered near 3.4s."""
    from cowp.geometry.lane_graph import trajectory_entry_to_region

    cfg = load_config("configs/label_cowp_v16_8.yaml")
    cfg["candidate"]["map_filter_enabled"] = False
    cfg["candidate"]["timing_envelope_gap_s"] = [0.8, 1.4]
    cfg["limits"]["max_candidates"] = 64
    steps, cur = 91, 10
    states = np.zeros((2, steps, 11), dtype=np.float32)
    for t in range(steps):
        tau = (t - cur) * 0.1
        states[0, t] = _state(10.0 * tau, 0.0, 10.0, 0.0, 0.0)
        states[1, t] = _state(30.0, -20.0 + 5.0 * tau, 0.0, 5.0, np.pi / 2.0)
    scene = ScenarioData(
        scenario_id="rmr-bcte-boundary-time",
        timestamps=np.arange(steps, dtype=np.float32) * 0.1,
        current_time_index=cur,
        states=states,
        object_type=np.asarray([ObjectType.VEHICLE, ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.asarray([0, 1], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.asarray([1], dtype=np.int64),
        tracks_to_predict=np.asarray([1], dtype=np.int32),
        map_data=MapData(),
    )
    region = ConflictRegion(3, "LANE_INTERSECTION", np.asarray([30.0, 0.0], dtype=np.float32), 3.0, (1, 2))
    out = generate_ego_candidates(scene, cfg, conflict_regions=[region])
    mask = (
        out["valid"]
        & (out["proposal_source"] == int(ProposalSource.ROBUST_BCTE))
        & (out["proposal_timing_side"] == 1)
    )
    idx = np.where(mask)[0]
    assert len(idx) > 0
    for k in idx:
        actual_tta, _ = trajectory_entry_to_region(
            out["trajectory"][k], region, current_state=states[0, cur], dt=0.1
        )
        target = float(out["proposal_target_time_s"][k])
        assert abs(actual_tta - target) <= cfg["candidate"]["timing_envelope_max_target_tta_error_s"] + 1e-6
        assert abs(float(out["proposal_target_tta_error_s"][k]) - abs(actual_tta - target)) < 1e-5


def test_unreachable_conflict_does_not_create_false_timing_proposals() -> None:
    cfg = load_config("configs/label_cowp_v16_8.yaml")
    cfg["candidate"]["map_filter_enabled"] = False
    steps, cur = 91, 10
    states = np.zeros((1, steps, 11), dtype=np.float32)
    for t in range(steps):
        tau = (t - cur) * 0.1
        states[0, t] = _state(10.0 * tau, 0.0, 10.0, 0.0, 0.0)
    scene = ScenarioData(
        scenario_id="conflict-behind-ego",
        timestamps=np.arange(steps, dtype=np.float32) * 0.1,
        current_time_index=cur,
        states=states,
        object_type=np.asarray([ObjectType.VEHICLE], dtype=np.int32),
        track_id=np.asarray([0], dtype=np.int64),
        sdc_track_index=0,
        objects_of_interest=np.asarray([], dtype=np.int64),
        tracks_to_predict=np.asarray([], dtype=np.int32),
        map_data=MapData(),
    )
    behind = ConflictRegion(7, "LANE_INTERSECTION", np.asarray([-20.0, 0.0], dtype=np.float32), 3.0, (1, 2))
    out = generate_ego_candidates(scene, cfg, conflict_regions=[behind])
    source = out["proposal_source"][out["valid"]]
    assert int(ProposalSource.ROBUST_BCTE) not in set(source.tolist())
    assert int(ProposalSource.LEGACY_TIMING) not in set(source.tolist())
    assert int(ProposalSource.STOP) not in set(source.tolist())


def test_online_bcs_timing_profile_matches_smooth_primitive() -> None:
    from cowp.label.trajectory_primitives import smooth_arrival_trajectory
    from cowp.waymax_eval.policy_wrapper import _online_smooth_arrival_profile

    cfg = load_config("configs/label_cowp_v16_8.yaml")
    current = _state(0.0, 0.0, 10.0, 0.0, 0.0)
    distance, target = 24.0, 4.5
    profile = _online_smooth_arrival_profile(distance, 10.0, target, cfg)
    assert profile is not None
    traj = smooth_arrival_trajectory(current, 80, 0.1, distance_m=distance, target_time_s=target)
    assert traj is not None
    # At the exact grid-aligned target time, position must realize the advertised
    # conflict-path distance without stopping early.
    k = int(round(target / 0.1)) - 1
    assert abs(float(traj[k, 0]) - distance) < 1e-4
    assert float(np.linalg.norm(traj[k, 3:5])) > 0.0


def test_priority_hold_release_is_smooth_valid_and_reaches_advertised_boundary_distance() -> None:
    from cowp.label.ego_candidates import _candidate_invalid_reason
    from cowp.label.trajectory_primitives import priority_hold_release_trajectory

    cfg = load_config("configs/label_cowp_v16_8.yaml")
    current = _state(0.0, 0.0, 10.0, 0.0, 0.0)
    target = 7.0
    traj = priority_hold_release_trajectory(
        current,
        80,
        0.1,
        entry_distance_m=20.0,
        target_time_s=target,
        stop_margin_m=3.0,
        release_speed_mps=4.0,
        min_hold_s=0.35,
    )
    assert traj is not None
    assert _candidate_invalid_reason(traj, cfg, None) is None
    k = int(round(target / 0.1)) - 1
    assert abs(float(traj[k, 0]) - 20.0) < 1e-4
    speed = np.linalg.norm(traj[:, 3:5], axis=-1)
    assert int((speed[:k] < 0.2).sum()) >= 3
    assert np.all(np.diff(traj[:, 0]) >= -1e-4)


def test_priority_hold_release_is_generated_for_protected_stop_control_interaction() -> None:
    from cowp.core.types import Lane

    cfg = load_config("configs/label_cowp_v16_8.yaml")
    cfg["candidate"]["map_filter_enabled"] = False
    cfg["limits"]["max_candidates"] = 96
    steps, cur = 91, 10
    states = np.zeros((2, steps, 11), dtype=np.float32)
    for t in range(steps):
        tau = (t - cur) * 0.1
        states[0, t] = _state(10.0 * tau, 0.0, 10.0, 0.0, 0.0)
        states[1, t] = _state(30.0, -30.0 + 5.0 * tau, 0.0, 5.0, np.pi / 2.0)
    map_data = MapData(lanes={
        1: Lane(1, np.asarray([[-20.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=np.float32), controlled_by_stop=True),
        2: Lane(2, np.asarray([[30.0, -60.0, 0.0], [30.0, 60.0, 0.0]], dtype=np.float32), controlled_by_stop=False),
    })
    scene = ScenarioData(
        scenario_id="priority-hold-release-toy",
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
    valid = out["valid"]
    phr = valid & (out["proposal_source"] == int(ProposalSource.PRIORITY_HOLD_RELEASE))
    assert phr.any()
    assert np.all(out["proposal_timing_side"][phr] == 1)
    assert np.all(out["proposal_target_agent_index"][phr] == 1)
    assert np.max(out["proposal_target_tta_error_s"][phr]) <= cfg["candidate"]["timing_envelope_max_target_tta_error_s"] + 1e-6
    # The core neutral option is reserved before optional interaction proposals.
    assert np.any(valid & (out["proposal_source"] == int(ProposalSource.NEUTRAL)))
