from __future__ import annotations

import numpy as np
import torch

from cowp.planning.set_preservation_selector import (
    select_set_preservation_frontier_1d,
    select_set_preservation_frontier_batch,
)
from cowp.waymax_eval.policy_wrapper import _one_step_action_risk_np


def _selector_inputs():
    return dict(
        scores=torch.tensor([0.2, 0.3, 0.5, 0.1]),
        base_mask=torch.tensor([True, True, True, True]),
        noncoercive_risk=torch.tensor([0.1, 0.2, 0.8, 0.3]),
        score_risk=torch.tensor([0.2, 0.3, 0.8, 0.1]),
        progress=torch.tensor([8.0, 7.0, 9.0, 5.0]),
        progress_shortfall=torch.tensor([0.1, 0.2, 0.0, 0.4]),
        action_risk=torch.tensor([0.1, 0.2, 0.9, 0.1]),
        rule_risk=torch.tensor([0.1, 0.2, 0.1, 0.2]),
        outcome_risk=torch.tensor([0.1, 0.2, 0.2, 0.1]),
        ncf_probability=torch.tensor([0.9, 0.8, 0.2, 0.7]),
        false_safe_probability=torch.tensor([0.1, 0.2, 0.8, 0.3]),
        cfg={
            "candidate_frontier_mode": "epsilon_pareto",
            "candidate_frontier_min_keep": 2,
            "candidate_frontier_max_keep": 3,
            "candidate_frontier_max_action_risk": 0.4,
            "candidate_selection_risk_budget": 0.3,
        },
    )


def test_shared_selector_batch_matches_single_scene():
    one = _selector_inputs()
    result = select_set_preservation_frontier_1d(**one)
    batched = {
        k: (v[None] if torch.is_tensor(v) else v)
        for k, v in one.items()
    }
    frontier, adjusted, _ = select_set_preservation_frontier_batch(**batched)
    assert torch.equal(frontier[0], result.frontier)
    assert torch.allclose(adjusted[0], result.adjusted_scores, equal_nan=True)
    assert result.frontier.sum() >= 1
    assert not result.frontier[2]  # high action risk is shielded


def test_vectorized_action_risk_returns_reusable_targets():
    cfg = {
        "time": {"dt": 0.1},
        "candidate": {
            "max_accel_mps2": 4.0,
            "max_decel_mps2": 6.0,
            "max_jerk_mps3": 10.0,
            "max_yaw_rate_rad_s": 1.2,
        },
        "planning": {
            "online_action_risk_horizon_steps": 8,
            "candidate_action_projection_risk_mix": 0.75,
        },
        "waymax": {"max_delta_yaw_rad": 0.12},
    }
    agents = np.zeros((4, 11), dtype=np.float32)
    agents[:, 10] = 1.0
    agents[0, 3] = agents[0, 5] = 5.0
    candidates = np.zeros((3, 12, 7), dtype=np.float32)
    candidates[:, :, 0] = np.arange(1, 13, dtype=np.float32)[None] * 0.5
    candidates[:, :, 3] = 5.0
    candidates[1, 0, 0] = 20.0  # large controller projection correction
    valid = np.array([True, True, False])
    risk, targets, accel = _one_step_action_risk_np(
        agents, 0, candidates, valid, cfg, return_targets=True
    )
    assert risk.shape == (3,)
    assert targets.shape == (3, 5)
    assert accel.shape == (3,)
    assert risk[1] > risk[0]
    assert risk[2] == 1.0
    assert np.isfinite(targets).all()


def test_planner_heads_do_not_backpropagate_into_witness_backbone():
    from cowp.models.cowp_model import COWPModel
    from cowp.waymax_eval.policy_wrapper import build_online_batch

    agents = np.zeros((5, 11), dtype=np.float32)
    agents[:, 7:10] = np.array([4.8, 1.9, 1.6], dtype=np.float32)
    agents[:, 10] = 1.0
    agents[0, 3] = agents[0, 5] = 5.0
    agents[1, 0:2] = [16.0, 0.5]
    cfg = {
        "model": {
            "d_state": 11, "history_steps": 11, "d_model": 32,
            "num_heads": 4, "num_layers": 1, "dropout": 0.0,
            "max_agents": 8, "max_natural_alternatives": 4,
            "max_safe_responses": 4, "future_steps": 10, "token_count": 7,
        },
        "limits": {
            "max_candidates": 16, "max_critical_agents": 4,
            "max_natural_alternatives": 4, "max_safe_responses": 4,
            "max_conflict_regions": 8, "max_agents": 8,
        },
        "time": {"future_steps": 10, "dt": 0.1},
        "candidate": {},
        "planning": {
            "planner_detach_witness_features": True,
            "planner_detach_backbone_features": True,
        },
        "ablation": {},
    }
    batch_np = build_online_batch(agents, 0, cfg)
    batch = {k: torch.as_tensor(v) for k, v in batch_np.items()}
    model = COWPModel(cfg)
    pred = model(batch, stage="planner")
    loss = (
        pred["planner_score"].sum()
        + pred["candidate_ncf_logit"].sum()
        + pred["candidate_false_safe_logit"].sum()
        + pred["outcome"]["collision_logit"].sum()
    )
    loss.backward()
    assert any(p.grad is not None for p in model.candidate_certificate.parameters())
    assert any(p.grad is not None for p in model.planner.parameters())
    assert all(p.grad is None for p in model.graph.parameters())
    assert all(p.grad is None for p in model.candidate_encoder.parameters())
    assert all(p.grad is None for p in model.witness_decoder.parameters())
