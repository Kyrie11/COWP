from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.waymax_eval.metrics_cowp import policy_diagnostic_scenario_rows
from cowp.waymax_eval.policy_wrapper import (
    _collision_free_against_constant_velocity,
    _lane_centerline_mask,
    _prepare_collision_check_context,
    _recovery_bridge_viability_mask,
    _route_lane_aware_candidates,
)


def _cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/label_cowp_v16_8.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("time", {})["future_steps"] = 80
    cfg["time"]["dt"] = 0.1
    return cfg


def _state(other_x: float = 25.0) -> np.ndarray:
    s = np.zeros((2, 11), dtype=np.float32)
    s[:, 7] = 4.8
    s[:, 8] = 1.9
    s[:, 9] = 1.6
    s[:, 10] = 1.0
    s[0] = np.asarray([0, 0, 0, 5, 0, 5, 0, 4.8, 1.9, 1.6, 1], dtype=np.float32)
    s[1] = np.asarray([other_x, 0, 0, 0, 0, 0, 0, 4.8, 1.9, 1.6, 1], dtype=np.float32)
    return s


def _empty_road() -> dict[str, np.ndarray]:
    return {
        "xy": np.zeros((0, 2), dtype=np.float32),
        "heading": np.zeros(0, dtype=np.float32),
        "valid": np.zeros(0, dtype=bool),
        "types": np.zeros(0, dtype=np.int32),
    }


def test_cached_conventional_screen_is_bit_exact_for_candidate_bank():
    cfg = _cfg()
    state = _state()
    road = _empty_road()
    H = int(cfg["time"]["future_steps"])
    context = _prepare_collision_check_context(state, 0, cfg, horizon=H)
    lane_mask = _lane_centerline_mask(road)

    legacy = _route_lane_aware_candidates(state, 0, road, cfg)
    cached = _route_lane_aware_candidates(
        state, 0, road, cfg, collision_context=context, lane_mask=lane_mask
    )
    assert len(legacy) == len(cached) == 5
    for lhs, rhs in zip(legacy, cached):
        np.testing.assert_array_equal(lhs, rhs)


def test_recovery_bridge_recovers_far_horizon_collision_without_relaxing_full_horizon_screen():
    cfg = _cfg()
    state = _state(other_x=25.0)
    road = _empty_road()
    H = int(cfg["time"]["future_steps"])
    candidate = constant_accel_trajectory(state[0], H, 0.1, accel=0.0)

    # The original 8 s primitive eventually reaches the stopped actor and must
    # remain outside the conventional set.
    assert not _collision_free_against_constant_velocity(candidate, state, 0, cfg)

    cands = np.zeros((2, H, 7), dtype=np.float32)
    cands[0] = candidate
    valid = np.asarray([True, False])
    conventional = np.asarray([False, False])
    bridge = _recovery_bridge_viability_mask(
        cands, valid, conventional, state, 0, road, cfg
    )
    # Receding-horizon commit + bounded-stop continuation avoids the far-future
    # collision while preserving the original full-horizon rejection above.
    assert bridge.tolist() == [True, False]


def test_recovery_bridge_never_promotes_invalid_or_already_conventional_candidate():
    cfg = _cfg()
    state = _state(other_x=25.0)
    road = _empty_road()
    H = int(cfg["time"]["future_steps"])
    tr = constant_accel_trajectory(state[0], H, 0.1, accel=0.0)
    cands = np.stack([tr, tr], axis=0)
    bridge = _recovery_bridge_viability_mask(
        cands,
        np.asarray([False, True]),
        np.asarray([False, True]),
        state,
        0,
        road,
        cfg,
    )
    assert not bridge.any()


def test_recovery_reason_is_counted_as_no_conventional_and_first_event_provenance():
    rollouts = [{
        "scenario_id": "recovery",
        "steps": 1,
        "policy_diagnostics": [{
            "fallback_used": True,
            "fallback_reason": "no_conventional_use_recovery_bridge",
            "selected_candidate_valid": True,
            "selected_candidate_conventional_safe": False,
            "selected_recovery_bridge_viable": True,
            "selected_macro_type": 1,
            "selected_macro_name": "YIELD",
            "valid_candidates": 10,
            "conventional_candidates": 0,
            "recovery_bridge_candidates": 3,
            "emergency_action_used": False,
            "execution_trajectory_source": "candidate",
        }],
        "standard_metrics": {
            "CR": 1.0,
            "CollisionRate": 1.0,
            "FirstPositiveStep/OverlapMetric": 1,
        },
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["no_conventional_step_rate"] == 1.0
    assert row["recovery_bridge_step_rate"] == 1.0
    assert row["recovery_bridge_available_step_rate"] == 1.0
    assert row["mean_recovery_bridge_candidates"] == 3.0
    assert row["selected_recovery_bridge_at_action_before_first_collision"] is True


def test_recovery_bridge_is_a_separate_selector_branch_not_a_cowp_main_path_change():
    from cowp.waymax_eval import policy_wrapper

    src = Path(policy_wrapper.__file__).read_text(encoding="utf-8")
    assert 'fallback_reason = "no_conventional_use_recovery_bridge"' in src
    # The new branch is after the full-horizon conventional fallback and before
    # unrestricted valid fallback.
    assert src.index('fallback_reason = "no_certificate_use_least_coercive_conventional"') < src.index(
        'fallback_reason = "no_conventional_use_recovery_bridge"'
    ) < src.index('fallback_reason = "no_conventional_use_least_coercive_valid"')
