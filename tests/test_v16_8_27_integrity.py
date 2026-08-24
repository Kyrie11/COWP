from __future__ import annotations

from pathlib import Path

import numpy as np


def test_online_candidate_api_has_no_conventional_bypass():
    import inspect
    from cowp.waymax_eval import policy_wrapper

    sig = inspect.signature(policy_wrapper._add_candidate)
    assert "conventional_check" not in sig.parameters
    src = Path(policy_wrapper.__file__).read_text(encoding="utf-8")
    assert "conventional_check=False" not in src


def test_neutral_candidate_is_retained_but_not_promoted_when_screen_fails(monkeypatch):
    from cowp.core.constants import MacroType
    from cowp.waymax_eval import policy_wrapper

    monkeypatch.setattr(policy_wrapper, "repair_planar_kinematics", lambda traj, *a, **k: traj)
    monkeypatch.setattr(policy_wrapper, "_candidate_dyn_ok", lambda traj, cfg: True)
    monkeypatch.setattr(policy_wrapper, "_roadgraph_drivable_mask", lambda traj, roadgraph: False)
    calls = {"collision": 0}

    def _collision(*args, **kwargs):
        calls["collision"] += 1
        return True

    monkeypatch.setattr(policy_wrapper, "_collision_free_against_constant_velocity", _collision)

    out, macros, utils, conventional = [], [], [], []
    traj = np.zeros((8, 7), dtype=np.float32)
    agent_state = np.zeros((1, 10), dtype=np.float32)
    cfg = {"limits": {"max_candidates": 4}, "time": {"dt": 0.1}, "planning": {}}
    policy_wrapper._add_candidate(
        out, macros, utils, conventional, traj, MacroType.NEUTRAL_EGO, 0.8,
        agent_state, 0, {}, cfg,
    )
    assert len(out) == 1
    assert macros == [int(MacroType.NEUTRAL_EGO)]
    assert conventional == [False]
    # Python short-circuiting is fine: the roadgraph audit failed, so collision
    # audit need not run.  The key contract is that the candidate is not promoted.
    assert calls["collision"] == 0


def test_neutral_candidate_enters_conventional_pool_only_after_both_screens_pass(monkeypatch):
    from cowp.core.constants import MacroType
    from cowp.waymax_eval import policy_wrapper

    monkeypatch.setattr(policy_wrapper, "repair_planar_kinematics", lambda traj, *a, **k: traj)
    monkeypatch.setattr(policy_wrapper, "_candidate_dyn_ok", lambda traj, cfg: True)
    monkeypatch.setattr(policy_wrapper, "_roadgraph_drivable_mask", lambda traj, roadgraph: True)
    monkeypatch.setattr(policy_wrapper, "_collision_free_against_constant_velocity", lambda *a, **k: True)

    out, macros, utils, conventional = [], [], [], []
    traj = np.zeros((8, 7), dtype=np.float32)
    agent_state = np.zeros((1, 10), dtype=np.float32)
    cfg = {"limits": {"max_candidates": 4}, "time": {"dt": 0.1}, "planning": {}}
    policy_wrapper._add_candidate(
        out, macros, utils, conventional, traj, MacroType.NEUTRAL_EGO, 0.8,
        agent_state, 0, {}, cfg,
    )
    assert conventional == [True]


def test_outcome_head_metadata_is_method_local():
    from cowp.waymax_eval.rollout import _outcome_head_selection_metadata

    assert _outcome_head_selection_metadata("cowp", 0.0) == (False, "none")
    assert _outcome_head_selection_metadata("cowp_fallback_outcome", 0.0) == (True, "fallback_only")
    assert _outcome_head_selection_metadata("cowp", 0.2) == (True, "all")


def test_first_event_provenance_keeps_macro_conventional_and_reason():
    from cowp.waymax_eval.metrics_cowp import policy_diagnostic_scenario_rows

    rollouts = [{
        "scenario_id": "s1",
        "steps": 3,
        "policy_diagnostics": [
            {
                "fallback_used": False,
                "fallback_reason": "accepted_priority_ncf",
                "selected_candidate_valid": True,
                "selected_candidate_conventional_safe": True,
                "selected_macro_type": 0,
                "selected_macro_name": "KEEP_LANE",
            },
            {
                "fallback_used": True,
                "fallback_reason": "no_certificate_use_least_coercive_conventional",
                "selected_candidate_valid": True,
                "selected_candidate_conventional_safe": True,
                "selected_macro_type": 1,
                "selected_macro_name": "YIELD",
            },
            {
                "fallback_used": True,
                "fallback_reason": "no_conventional_use_least_coercive_valid",
                "selected_candidate_valid": True,
                "selected_candidate_conventional_safe": False,
                "selected_macro_type": 11,
                "selected_macro_name": "NEUTRAL_EGO",
            },
        ],
        # FirstPositiveStep is 1-indexed after the action/environment step.  A
        # value of 2 therefore attributes the first event to policy row index 1.
        "standard_metrics": {
            "CR": 1.0,
            "CollisionRate": 1.0,
            "FirstPositiveStep/OverlapMetric": 2,
        },
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["fallback_at_action_before_first_collision"] is True
    assert row["selected_conventional_safe_at_action_before_first_collision"] is True
    assert row["selected_macro_name_at_action_before_first_collision"] == "YIELD"
    assert row["fallback_reason_at_action_before_first_collision"] == "no_certificate_use_least_coercive_conventional"
