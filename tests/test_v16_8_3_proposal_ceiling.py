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
        states[1, t] = _state(30.0, -20.0 + 5.0 * tau, 0.0, 5.0, np.pi / 2.0)
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
    assert metrics["ProposalDiagnostics/Version"] == "v16_8_3_proposal_floor"
