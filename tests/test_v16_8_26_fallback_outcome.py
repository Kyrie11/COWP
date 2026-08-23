from __future__ import annotations

import pytest


def _case(torch, *, reject_all: bool):
    traj = torch.zeros((1, 2, 6, 7), dtype=torch.float32)
    traj[0, 0, :, 0] = torch.linspace(0.0, 5.0, 6)
    traj[0, 1, :, 0] = torch.linspace(0.0, 5.0, 6)
    batch = {
        "cowp/candidates/valid": torch.ones((1, 2), dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones((1, 2), dtype=torch.bool),
        "cowp/candidates/ego_utility_prior": torch.tensor([[0.0, 1.0]]),
        "cowp/candidates/trajectory": traj,
        "cowp/candidates/rule_risk": torch.zeros((1, 2)),
        "cowp/critical/valid": torch.ones((1, 1), dtype=torch.bool),
        "cowp/witness/rho": torch.full((1, 2, 1), 2, dtype=torch.long),
    }
    risk = 0.9 if reject_all else 0.1
    pred = {
        "planner_score": torch.tensor([[0.0, 1.0]]),
        "witness": {"exist_logits": torch.full((1, 2, 1), -8.0), "opr": torch.ones((1, 2, 1))},
        "candidate_transport_risk": torch.tensor([[risk, risk]]),
        "candidate_transport_uncertainty": torch.zeros((1, 2)),
        "candidate_transport_severe_prob": torch.zeros((1, 2)),
        "outcome": {
            # candidate 0 is strongly unsafe; candidate 1 is low risk
            "collision_logit": torch.tensor([[8.0, -8.0]]),
            "offroad_logit": torch.tensor([[8.0, -8.0]]),
        },
    }
    cfg = {
        "time": {"dt": 0.1},
        "planning": {
            "candidate_transport_pure_selector": True,
            "candidate_transport_gate_mode": "budget",
            "candidate_transport_budget": 0.5,
            "candidate_transport_ucb_scale": 0.0,
            "candidate_transport_uncertainty_penalty": 0.0,
            "candidate_frontier_keep_fraction": 1.0,
            "candidate_frontier_min_keep": 2,
            "candidate_frontier_max_keep": 2,
            "candidate_selection_risk_budget": 1.0,
            "candidate_selection_outcome_weight": 0.0,
            "candidate_frontier_shield_tie_mix": 0.0,
            "candidate_min_ncf_prob": 0.0,
            "candidate_max_false_safe_prob": 1.0,
            "candidate_hard_max_action_risk": 1.0,
            "candidate_hard_max_rule_risk": 1.0,
            "witness_probability_source": "logit",
            "fallback_transport_weight": 0.0,
            "fallback_rule_weight": 0.0,
            "fallback_action_weight": 0.0,
            "fallback_pressure_weight": 0.0,
            "fallback_outcome_weight": 1.0,
            "fallback_utility_weight": 0.05,
            "fallback_stop_like_bonus": 0.0,
        },
    }
    return batch, pred, cfg


def test_fallback_outcome_is_identical_to_cowp_when_certificate_accepts():
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.rollout import _select_from_learned
    batch, pred, cfg = _case(torch, reject_all=False)
    base = _select_from_learned(batch, pred, method="cowp", gate_mode="priority", witness_threshold=.7, bcot_risk_budget=.5, cfg=cfg)
    guard = _select_from_learned(batch, pred, method="cowp_fallback_outcome", gate_mode="priority", witness_threshold=.7, bcot_risk_budget=.5, cfg=cfg)
    assert base[0] == guard[0]
    assert base[1][0].tolist() == guard[1][0].tolist()
    assert base[2][0].tolist() == guard[2][0].tolist()
    assert base[3] == guard[3] == [False]


def test_fallback_outcome_changes_only_uncertified_fallback_ranking():
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.rollout import _select_from_learned
    batch, pred, cfg = _case(torch, reject_all=True)
    base = _select_from_learned(batch, pred, method="cowp", gate_mode="priority", witness_threshold=.7, bcot_risk_budget=.5, cfg=cfg)
    guard = _select_from_learned(batch, pred, method="cowp_fallback_outcome", gate_mode="priority", witness_threshold=.7, bcot_risk_budget=.5, cfg=cfg)
    assert base[1][0].tolist() == guard[1][0].tolist() == [False, False]
    assert base[3] == guard[3] == [True]
    assert base[0] == [0]
    assert guard[0] == [1]


def test_fallback_outcome_method_keeps_priority_gate_offline_and_online():
    from cowp.waymax_eval.rollout import _method_gate_defaults
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method
    assert _method_gate_defaults("cowp_fallback_outcome", "hard") == ("cowp_fallback_outcome", "priority")
    assert _canonical_online_method("cowp_fallback_outcome", "hard") == ("cowp_fallback_outcome", "priority")


def test_failure_attribution_uses_first_event_prefix():
    from cowp.waymax_eval.metrics_cowp import physical_failure_attribution_summary, policy_diagnostic_scenario_rows
    rollouts = [{
        "scenario_id": "s1", "steps": 4,
        "policy_diagnostics": [
            {"fallback_used": False, "fallback_reason": "accepted_priority_ncf"},
            {"fallback_used": True, "fallback_reason": "no_certificate_use_least_coercive_conventional"},
            {"fallback_used": True, "fallback_reason": "no_certificate_use_least_coercive_conventional"},
            {"fallback_used": False, "fallback_reason": "accepted_priority_ncf"},
        ],
        "standard_metrics": {"CR": 1.0, "CollisionRate": 1.0, "OffroadRate": 0.0, "FirstPositiveStep/OverlapMetric": 3},
    }]
    row = policy_diagnostic_scenario_rows(rollouts)[0]
    assert row["fallback_step_rate"] == pytest.approx(0.5)
    assert row["fallback_rate_before_first_collision"] == pytest.approx(2/3)
    summary = physical_failure_attribution_summary(rollouts)
    assert summary["Collision/MeanFallbackRateBeforeFirstEvent"] == pytest.approx(2/3)
