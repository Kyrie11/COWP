from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cowp.external_baselines.adapters import build_gameformer_map, make_external_batch
from cowp.external_baselines.dtpp_cowp import dtpp_loss
from cowp.external_baselines.gameformer_cowp import gameformer_loss
from cowp.external_baselines.plant2_cowp import plant2_loss
from cowp.external_baselines.pluto_cowp import pluto_loss
from cowp.external_baselines.rule_based import rule_costs_for_batch


def test_external_trainer_recovers_fp32_norm_overflow_without_hiding_bad_entries():
    train_mod = importlib.import_module("cowp.scripts.20_train_external_baseline")
    model = nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 3.0e38)
    assert torch.isfinite(model.weight.grad).all()
    norm, bad, used_fp64 = train_mod._clip_grad_norm_stable(model, 1.0)
    assert bad == []
    assert used_fp64 is True
    assert torch.isfinite(norm)
    assert float(norm) > 3.0e38
    assert torch.isfinite(model.weight.grad).all()
    assert float(torch.linalg.vector_norm(model.weight.grad.float())) <= 1.0001

    model.weight.grad = torch.tensor([[float("inf"), 1.0]], dtype=model.weight.dtype)
    norm2, bad2, used_fp642 = train_mod._clip_grad_norm_stable(model, 1.0)
    assert not torch.isfinite(norm2)
    assert bad2 == ["weight"]
    assert used_fp642 is False


def _invalid_sdc_batch(T: int = 4):
    B, N, H, K = 1, 2, 11, 2
    hist = torch.zeros(B, N, H, 11)
    hist[..., 10] = 1.0
    hist[0, 0, -1, 0] = float("nan")
    is_sdc = torch.tensor([[1.0, 0.0]])
    cand = torch.zeros(B, K, T, 7)
    return {
        "state/history": hist,
        "state/is_sdc": is_sdc,
        "state/type": torch.ones(B, N),
        "state/future/x": torch.zeros(B, N, T),
        "state/future/y": torch.zeros(B, N, T),
        "state/future/valid": torch.ones(B, N, T),
        "cowp/candidates/trajectory": cand,
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
        "roadgraph_samples/xyz": torch.zeros(B, 8, 3),
        "roadgraph_samples/valid": torch.ones(B, 8, dtype=torch.bool),
    }


def test_adapter_invalidates_nonfinite_declared_valid_sdc_and_keeps_model_inputs_finite():
    T = 4
    cfg = {"limits": {"max_agents": 8}, "model": {"history_steps": 11}, "time": {"future_steps": T}}
    ext = make_external_batch(
        _invalid_sdc_batch(T), cfg, device=torch.device("cpu"), max_neighbors=1,
        max_candidates=2, horizon=T, baseline=None, require_candidates=True, require_future=True,
    )
    assert ext.sdc_current_valid.tolist() == [False]
    assert not bool(ext.ego_future_valid.any())
    assert not bool(ext.candidate_valid.any())
    for family in (ext.gameformer_inputs, ext.dtpp_inputs, ext.planner_inputs):
        for value in family.values():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                assert torch.isfinite(value).all()


def test_marked_valid_nonfinite_roadgraph_point_is_masked():
    xyz = torch.zeros(1, 4, 3)
    xyz[0, 1, 0] = float("nan")
    batch = {"roadgraph_samples/xyz": xyz, "roadgraph_samples/valid": torch.ones(1, 4, dtype=torch.bool)}
    _, _, valid, _ = build_gameformer_map(
        batch, 1, torch.device("cpu"), origin=torch.zeros(1, 2), yaw0=torch.zeros(1), return_valid=True
    )
    assert bool(valid[0, 0, 0, 0])
    assert not bool(valid[0, 0, 0, 1])




def test_marked_valid_nonfinite_split_xy_roadgraph_point_is_masked():
    batch = {
        "roadgraph_samples/x": torch.tensor([[0.0, float("nan"), 2.0, 3.0]]),
        "roadgraph_samples/y": torch.zeros(1, 4),
        "roadgraph_samples/valid": torch.ones(1, 4, dtype=torch.bool),
    }
    _, _, valid, _ = build_gameformer_map(
        batch, 1, torch.device("cpu"), origin=torch.zeros(1, 2), yaw0=torch.zeros(1), return_valid=True
    )
    assert bool(valid[0, 0, 0, 0])
    assert not bool(valid[0, 0, 0, 1])

def test_gameformer_masked_nan_label_does_not_contaminate_loss_or_metrics():
    B, N, M, T = 1, 1, 2, 3
    gmm = torch.zeros(B, N, M, T, 4, requires_grad=True)
    scores = torch.zeros(B, N, M, requires_grad=True)
    outputs = {"level_0_interactions": gmm, "level_0_scores": scores}
    ego = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")], [1.0, 0.0]]])
    valid = torch.tensor([[True, False, True]])
    loss, metrics = gameformer_loss(
        outputs, ego, valid, torch.empty(B, 0, T, 2), torch.empty(B, 0, T, dtype=torch.bool)
    )
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["plannerADE"])
    loss.backward()
    assert torch.isfinite(gmm.grad).all()
    assert torch.isfinite(scores.grad).all()


class _DummyDTPP(nn.Module):
    def forward(self, inputs, ego_traj_tree, timesteps=80, candidate_valid=None):
        B, K, T = ego_traj_tree.shape[:3]
        A = 1
        base = ego_traj_tree[..., :1].sum() * 0.0
        neighbors = base + torch.zeros(B, K, A, T, 3, requires_grad=True)
        scores = base + torch.zeros(B, K, requires_grad=True)
        ego_reg = base + torch.zeros(B, T, 2, requires_grad=True)
        weights = base + torch.ones(B, 8, requires_grad=True)
        return neighbors, scores, ego_reg, weights


class _DummyPLUTO(nn.Module):
    def forward(self, inputs):
        B, T = 1, 3
        return {
            "trajectories": torch.zeros(B, 2, T, 2, requires_grad=True),
            "scores": torch.zeros(B, 2, requires_grad=True),
            "aux_trajectory": torch.zeros(B, T, 2, requires_grad=True),
            "scene_embedding": torch.zeros(B, 4, requires_grad=True),
        }


class _DummyPlanT(nn.Module):
    def forward(self, inputs):
        B, T = 1, 3
        return {
            "trajectory": torch.zeros(B, T, 2, requires_grad=True),
            "speed": torch.zeros(B, T, requires_grad=True),
            "hazard_logit": torch.zeros(B, requires_grad=True),
            "scene_embedding": torch.zeros(B, 4, requires_grad=True),
        }


def test_other_learned_losses_mask_nan_targets_before_nonlinear_arithmetic():
    T, K = 3, 2
    ego = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")], [1.0, 0.0]]])
    valid = torch.tensor([[True, False, True]])
    neigh = torch.tensor([[[[0.0, 0.0], [float("nan"), float("nan")], [2.0, 0.0]]]])
    neigh_valid = torch.tensor([[[True, False, True]]])
    tree = torch.zeros(1, K, T, 6)
    candidate_valid = torch.tensor([[True, True]])
    loss_d, metrics_d = dtpp_loss(
        _DummyDTPP(), {}, tree, candidate_valid, torch.tensor([0]), ego, valid, neigh, neigh_valid, timesteps=T
    )
    assert torch.isfinite(loss_d)
    assert np.isfinite(metrics_d["plannerADE"])
    loss_d.backward()

    loss_p, metrics_p = pluto_loss(_DummyPLUTO(), {}, ego, valid, contrast_weight=0.0)
    assert torch.isfinite(loss_p)
    assert np.isfinite(metrics_p["plannerADE"])
    loss_p.backward()

    plan_inputs = {"neighbors_future_xy": neigh, "neighbors_future_valid": neigh_valid}
    loss_t, metrics_t = plant2_loss(_DummyPlanT(), plan_inputs, ego, valid)
    assert torch.isfinite(loss_t)
    assert np.isfinite(metrics_t["plannerADE"])
    loss_t.backward()


def test_rule_baseline_invalid_sdc_state_produces_no_selectable_candidate():
    cand = np.zeros((1, 2, 4, 7), dtype=np.float32)
    hist = np.zeros((1, 1, 11, 11), dtype=np.float32)
    hist[..., 10] = 1.0
    hist[0, 0, -1, 0] = np.nan
    batch = {
        "cowp/candidates/trajectory": cand,
        "cowp/candidates/valid": np.ones((1, 2), dtype=bool),
        "cowp/candidates/conventional_safe": np.ones((1, 2), dtype=bool),
        "state/history": hist,
        "state/is_sdc": np.ones((1, 1), dtype=bool),
    }
    _, accept, valid_out = rule_costs_for_batch(batch, {}, "pdm_closed", require_conventional_safe=False)
    assert not bool(valid_out.any())
    assert not bool(accept.any())


def test_external_waymax_policies_use_nonpadding_execution_resolver():
    root = Path(__file__).resolve().parents[1]
    for rel in ("cowp/external_baselines/waymax_policy.py", "cowp/external_baselines/rule_waymax_policy.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "_resolve_execution_trajectory(" in text
        assert "if valid_idx.size else 0" not in text
    learned_text = (root / "cowp/external_baselines/waymax_policy.py").read_text(encoding="utf-8")
    assert "valid = valid & candidate_finite & adapter_valid" in learned_text
