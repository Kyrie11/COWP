from __future__ import annotations

import torch

from cowp.data.dataset import _wanted_keys_for_stage
from cowp.models.losses import response_loss, set_transport_loss
from cowp.models.response_decoder import ResponseDecoder


def test_planner_loads_compact_response_and_transport_targets():
    keys = _wanted_keys_for_stage(
        "planner", include_response_traj=False,
        include_response_components=False, include_waymax_outcomes=False,
    )
    assert keys is not None
    assert "cowp/response/valid" in keys
    assert "cowp/response/source" in keys
    assert "cowp/transport/" in keys
    assert "cowp/response/traj" not in keys


def test_compact_response_predicts_same_root_assignment():
    dec = ResponseDecoder(d_model=8, responses=3, future_steps=4, natural_modes=5)
    out = dec(
        torch.randn(1, 4, 8), torch.randn(1, 2, 8), torch.randn(1, 8),
        torch.tensor([[1]]), decode_traj=False,
    )
    assert out["root_logits"].shape == (1, 2, 1, 3, 5)
    assert out["min_burden_logits"].shape == (1, 2, 1, 3)
    assert "traj" not in out


def test_root_and_mode_losses_are_active():
    B, K, A, R, M = 1, 2, 1, 3, 4
    response_pred = {
        "safe_logits": torch.randn(B, K, A, R),
        "low_logits": torch.randn(B, K, A, R),
        "valid_logits": torch.randn(B, K, A, R),
        "source_logits": torch.randn(B, K, A, R, 4),
        "root_logits": torch.randn(B, K, A, R, M),
        "min_burden_logits": torch.randn(B, K, A, R),
        "burden_total": torch.rand(B, K, A, R),
        "burden_components": torch.rand(B, K, A, R, 6),
    }
    batch = {
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/response/valid": torch.ones(B, K, A, R, dtype=torch.bool),
        "cowp/response/is_safe": torch.randint(0, 2, (B, K, A, R)).bool(),
        "cowp/response/is_low_burden": torch.randint(0, 2, (B, K, A, R)).bool(),
        "cowp/response/source": torch.zeros(B, K, A, R, dtype=torch.long),
        "cowp/response/burden_total": torch.rand(B, K, A, R),
        "cowp/transport/response_root_index": torch.randint(0, M, (B, K, A, R)),
        "cowp/transport/response_is_min_burden": torch.randint(0, 2, (B, K, A, R)).bool(),
    }
    rl = response_loss(response_pred, batch, {"response_traj_l1": 0.0, "response_components_l1": 0.0})
    assert torch.isfinite(rl["loss"])
    assert rl["root"].item() > 0

    set_pred = {
        "exist_logits": torch.randn(B, K, A),
        "opr": torch.rand(B, K, A),
        "min_safe_burden": torch.rand(B, K, A),
        "natural_conflict_mass": torch.rand(B, K, A),
        "response_exist_low_safe": torch.rand(B, K, A).clamp(0.01, 0.99),
        "mode_conflict_prob": torch.rand(B, K, A, M).clamp(0.01, 0.99),
        "mode_retain_prob": torch.rand(B, K, A, M).clamp(0.01, 0.99),
        "mode_uncertainty": torch.rand(B, K, A, M),
        "root_recovery_mass": torch.rand(B, K, A),
    }
    batch.update({
        "cowp/witness/exists": torch.randint(0, 2, (B, K, A)).bool(),
        "cowp/witness/opr": torch.rand(B, K, A),
        "cowp/witness/burden_total": torch.rand(B, K, A),
        "cowp/witness/natural_conflict_mass": torch.rand(B, K, A),
        "cowp/transport/mode_valid": torch.ones(B, K, A, M, dtype=torch.bool),
        "cowp/transport/mode_conflict": torch.randint(0, 2, (B, K, A, M)).bool(),
        "cowp/transport/mode_retained_low_safe": torch.randint(0, 2, (B, K, A, M)).bool(),
        "cowp/transport/root_recovery_mass": torch.rand(B, K, A),
    })
    sl = set_transport_loss(set_pred, batch, {})
    assert torch.isfinite(sl["loss"])
    assert sl["mode_conflict"].item() > 0
    assert sl["mode_retain"].item() > 0


def test_unordered_natural_modes_are_aligned_before_root_supervision():
    from cowp.models.losses import _gt_to_pred_natural_assignment, _align_pred_modes_to_gt

    # GT mode 0 is the x=0 trajectory and GT mode 1 is x=10. Predicted order is swapped.
    gt = torch.zeros(1, 1, 2, 3, 5)
    gt[:, :, 1, :, 0] = 10.0
    pred_traj = torch.zeros_like(gt)
    pred_traj[:, :, 0, :, 0] = 10.0
    pred_traj[:, :, 1, :, 0] = 0.0
    pred_dict = {"_natural_pred_traj": pred_traj}
    batch = {
        "cowp/natural/traj": gt,
        "cowp/natural/valid": torch.ones(1, 1, 2, dtype=torch.bool),
    }
    assignment = _gt_to_pred_natural_assignment(pred_dict, batch)
    assert assignment is not None
    assert assignment.tolist() == [[[1, 0]]]

    values = torch.tensor([[[[0.2, 0.8]]]])  # [B,K,A,M_pred]
    aligned = _align_pred_modes_to_gt(values, assignment)
    assert torch.allclose(aligned, torch.tensor([[[[0.8, 0.2]]]]))
