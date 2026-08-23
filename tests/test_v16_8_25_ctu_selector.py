from __future__ import annotations

import pytest


def _tiny_selector_case(torch):
    traj = torch.zeros((1, 2, 6, 7), dtype=torch.float32)
    traj[0, 0, :, 0] = torch.linspace(0.0, 4.0, 6)
    traj[0, 1, :, 0] = torch.linspace(0.0, 5.0, 6)
    batch = {
        "cowp/candidates/valid": torch.ones((1, 2), dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones((1, 2), dtype=torch.bool),
        "cowp/candidates/ego_utility_prior": torch.tensor([[1.0, 0.0]]),
        "cowp/candidates/trajectory": traj,
        "cowp/candidates/rule_risk": torch.zeros((1, 2)),
        "cowp/critical/valid": torch.ones((1, 1), dtype=torch.bool),
        "cowp/witness/rho": torch.full((1, 2, 1), 2, dtype=torch.long),
    }
    pred = {
        # candidate 1 is planner-preferred, candidate 0 is lower BCOT risk
        "planner_score": torch.tensor([[1.0, 0.0]]),
        "witness": {
            "exist_logits": torch.full((1, 2, 1), -8.0),
            "opr": torch.ones((1, 2, 1)),
        },
        "candidate_transport_risk": torch.tensor([[0.10, 0.30]]),
        "candidate_transport_uncertainty": torch.zeros((1, 2)),
        "candidate_transport_severe_prob": torch.zeros((1, 2)),
    }
    cfg = {
        "time": {"dt": 0.1},
        "planning": {
            "candidate_transport_pure_selector": True,
            "candidate_transport_gate_mode": "budget",
            "candidate_transport_budget": 0.50,
            "candidate_transport_ucb_scale": 0.0,
            "candidate_transport_uncertainty_penalty": 0.0,
            "candidate_frontier_keep_fraction": 1.0,
            "candidate_frontier_min_keep": 2,
            "candidate_frontier_max_keep": 2,
            "candidate_selection_risk_budget": 0.12,
            "candidate_min_ncf_prob": 0.0,
            "candidate_max_false_safe_prob": 1.0,
            "candidate_hard_max_action_risk": 1.0,
            "candidate_hard_max_rule_risk": 1.0,
            "witness_probability_source": "logit",
        },
    }
    return batch, pred, cfg


def test_ctu_keeps_same_certificate_but_removes_second_bcot_ranking():
    torch = pytest.importorskip("torch")
    from cowp.waymax_eval.rollout import _select_from_learned

    batch, pred, cfg = _tiny_selector_case(torch)
    ctu_sel, ctu_cert, ctu_short, _ = _select_from_learned(
        batch, pred, method="cowp_cert_utility", gate_mode="priority",
        witness_threshold=0.70, bcot_risk_budget=0.50, cfg=cfg,
    )
    cowp_sel, cowp_cert, _, _ = _select_from_learned(
        batch, pred, method="cowp", gate_mode="priority",
        witness_threshold=0.70, bcot_risk_budget=0.50, cfg=cfg,
    )

    assert ctu_cert[0].tolist() == cowp_cert[0].tolist() == [True, True]
    assert ctu_short[0].tolist() == [True, True]
    assert ctu_sel == [1]  # planner-score argmin after the hard certificate
    assert cowp_sel == [0]  # legacy frontier favors the lower BCOT-risk candidate


def test_ctu_method_is_priority_aware_offline_and_online():
    from cowp.waymax_eval.rollout import _method_gate_defaults
    from cowp.waymax_eval.policy_wrapper import _canonical_online_method

    assert _method_gate_defaults("cowp_cert_utility", "hard") == ("cowp_cert_utility", "priority")
    assert _canonical_online_method("cowp_cert_utility", "hard") == ("cowp_cert_utility", "priority")
