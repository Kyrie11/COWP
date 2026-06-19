from __future__ import annotations

import torch

from cowp.models.losses import natural_loss


def _batch_from_gt(gt: torch.Tensor, weight: torch.Tensor) -> dict[str, torch.Tensor]:
    B, A, M, T, D = gt.shape
    return {
        "cowp/natural/valid": torch.ones(B, A, M, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/natural/traj": gt,
        "cowp/natural/source": torch.zeros(B, A, M, dtype=torch.long),
        "cowp/natural/weight": weight,
        "cowp/natural/priority_preserved": torch.ones(B, A, M, dtype=torch.bool),
    }


def test_natural_loss_is_set_order_robust_for_trajectory_term():
    B, A, M, T, D = 1, 1, 3, 4, 7
    gt = torch.zeros(B, A, M, T, D)
    gt[:, :, 1, :, 0] = 10.0
    gt[:, :, 2, :, 1] = 20.0
    pred = {
        "traj": gt.clone(),
        "logits": torch.zeros(B, A, M),
        "source_logits": torch.zeros(B, A, M, 4),
        "priority_logits": torch.zeros(B, A, M),
    }
    weight = torch.ones(B, A, M) / M
    base = natural_loss(pred, _batch_from_gt(gt, weight), {"natural_traj_l1": 1.0, "natural_mode_ce": 0.0, "branch_source_ce": 0.0, "branch_minade": 0.0, "priority_preservation": 0.0, "neutral_consistency": 0.0, "diversity_loss": 0.0})
    perm = torch.tensor([2, 0, 1])
    gt_perm = gt[:, :, perm]
    weight_perm = weight[:, :, perm]
    permuted = natural_loss(pred, _batch_from_gt(gt_perm, weight_perm), {"natural_traj_l1": 1.0, "natural_mode_ce": 0.0, "branch_source_ce": 0.0, "branch_minade": 0.0, "priority_preservation": 0.0, "neutral_consistency": 0.0, "diversity_loss": 0.0})
    assert base["traj"].item() == 0.0
    assert permuted["traj"].item() == 0.0


def test_natural_source_distribution_loss_is_amp_dtype_safe():
    B, A, M, T, D = 1, 2, 3, 2, 7
    gt = torch.zeros(B, A, M, T, D)
    weight = torch.ones(B, A, M, dtype=torch.float32) / M
    batch = _batch_from_gt(gt, weight)
    batch["cowp/natural/source"] = torch.tensor([[[0, 1, 2], [1, 2, 0]]], dtype=torch.long)
    pred = {
        "traj": torch.zeros(B, A, M, T, D, dtype=torch.float16),
        "logits": torch.zeros(B, A, M, dtype=torch.float16),
        "source_logits": torch.zeros(B, A, M, 4, dtype=torch.float16),
        "priority_logits": torch.zeros(B, A, M, dtype=torch.float16),
    }
    losses = natural_loss(pred, batch, {
        "natural_traj_l1": 0.0,
        "natural_mode_ce": 0.0,
        "branch_source_ce": 1.0,
        "branch_minade": 0.0,
        "priority_preservation": 0.0,
        "neutral_consistency": 0.0,
        "diversity_loss": 0.0,
    })
    assert torch.isfinite(losses["source"])
    assert torch.isfinite(losses["loss"])
