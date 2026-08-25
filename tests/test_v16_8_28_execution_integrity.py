from __future__ import annotations

from pathlib import Path

import numpy as np


def _ego_state(x: float = 100.0, y: float = 50.0, speed: float = 8.0, yaw: float = 0.4) -> np.ndarray:
    # Online agent_state convention used by the policy wrapper:
    # x, y, z, vx, vy, speed, yaw, length, width, height.
    s = np.zeros(10, dtype=np.float32)
    s[0] = x
    s[1] = y
    s[5] = speed
    s[6] = yaw
    s[7] = 4.8
    s[8] = 1.9
    s[9] = 1.6
    return s


def test_no_valid_execution_never_consumes_zero_padding():
    from cowp.waymax_eval.policy_wrapper import _resolve_execution_trajectory

    padded = np.zeros((8, 20, 7), dtype=np.float32)
    current = _ego_state()
    cfg = {"time": {"dt": 0.1}, "planning": {"fallback_decel_mps2": -2.0}}

    traj, emergency, source = _resolve_execution_trajectory(padded, 0, False, current, cfg)
    assert emergency is True
    assert source == "bounded_smooth_stop"
    assert traj.shape == (20, 7)
    assert np.isfinite(traj).all()
    # A padded slot would be at the world origin.  The repair must stay anchored
    # to the current ego state instead.
    assert np.linalg.norm(traj[0, :2] - current[:2]) < 2.0
    assert np.linalg.norm(traj[0, :2]) > 50.0
    speeds = np.linalg.norm(traj[:, 3:5], axis=-1)
    assert np.all(np.diff(speeds) <= 1e-5)


def test_valid_execution_still_returns_exact_selected_candidate():
    from cowp.waymax_eval.policy_wrapper import _resolve_execution_trajectory

    cands = np.zeros((3, 5, 7), dtype=np.float32)
    cands[2, :, 0] = np.arange(5, dtype=np.float32) + 7.0
    traj, emergency, source = _resolve_execution_trajectory(cands, 2, True, _ego_state(), {"time": {"dt": 0.1}})
    assert emergency is False
    assert source == "candidate"
    np.testing.assert_array_equal(traj, cands[2])


def test_dead_emergency_stop_like_selector_branch_is_removed():
    from cowp.waymax_eval import policy_wrapper

    src = Path(policy_wrapper.__file__).read_text(encoding="utf-8")
    assert 'fallback_reason = "emergency_stop_like"' not in src


def test_emergency_execution_provenance_is_kept_at_first_event():
    from cowp.waymax_eval.metrics_cowp import policy_diagnostic_scenario_rows

    rollouts = [{
        "scenario_id": "s_emergency",
        "steps": 2,
        "policy_diagnostics": [
            {
                "fallback_used": False,
                "fallback_reason": "accepted_priority_ncf",
                "selected_candidate_valid": True,
                "selected_candidate_conventional_safe": True,
                "selected_macro_type": 0,
                "selected_macro_name": "KEEP_LANE",
                "valid_candidates": 4,
                "conventional_candidates": 2,
                "emergency_action_used": False,
                "execution_trajectory_source": "candidate",
            },
            {
                "fallback_used": True,
                "fallback_reason": "no_valid_candidate",
                "selected_candidate_valid": False,
                "selected_candidate_conventional_safe": False,
                "selected_macro_type": 11,
                "selected_macro_name": "EMERGENCY_BOUNDED_STOP",
                "valid_candidates": 0,
                "conventional_candidates": 0,
                "emergency_action_used": True,
                "execution_trajectory_source": "bounded_smooth_stop",
            },
        ],
        "standard_metrics": {
            "CR": 1.0,
            "OffroadRate": 1.0,
            "FirstPositiveStep/OffroadMetric": 2,
        },
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["emergency_action_step_rate"] == 0.5
    assert row["zero_valid_candidate_step_rate"] == 0.5
    assert row["zero_conventional_candidate_step_rate"] == 0.5
    assert row["emergency_action_at_action_before_first_offroad"] is True
    assert row["execution_trajectory_source_at_action_before_first_offroad"] == "bounded_smooth_stop"
