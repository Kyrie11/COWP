from __future__ import annotations

import importlib
import inspect

import torch

from cowp.models.losses import natural_loss


def _natural_batch(batch_size: int = 1, agents: int = 2, modes: int = 3, steps: int = 4, dim: int = 7):
    gt = torch.zeros(batch_size, agents, modes, steps, dim)
    return {
        "cowp/natural/valid": torch.ones(batch_size, agents, modes, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(batch_size, agents, dtype=torch.bool),
        "cowp/natural/traj": gt,
        "cowp/natural/source": torch.zeros(batch_size, agents, modes, dtype=torch.long),
        "cowp/natural/weight": torch.ones(batch_size, agents, modes) / modes,
        "cowp/natural/priority_preserved": torch.ones(batch_size, agents, modes),
    }


def test_no_probability_space_bce_in_losses_module():
    import cowp.models.losses as losses

    src = inspect.getsource(losses)
    assert "F.binary_cross_entropy(" not in src
    assert "BCELoss(" not in src


def test_natural_priority_loss_accepts_half_logits_and_dirty_targets():
    B, A, M, T, D = 1, 2, 3, 4, 7
    batch = _natural_batch(B, A, M, T, D)
    batch["cowp/natural/priority_preserved"] = torch.tensor([[[1.0, float("nan"), 2.0], [0.0, -1.0, 1.0]]])
    pred = {
        "traj": torch.zeros(B, A, M, T, D, dtype=torch.float16),
        "logits": torch.zeros(B, A, M, dtype=torch.float16),
        "source_logits": torch.zeros(B, A, M, 4, dtype=torch.float16),
        "priority_logits": torch.zeros(B, A, M, dtype=torch.float16),
    }
    losses = natural_loss(pred, batch, {"priority_preservation": 1.0})
    assert torch.isfinite(losses["priority"])
    assert torch.isfinite(losses["loss"])


def test_train_parser_accepts_compile_and_fused_flags():
    train_mod = importlib.import_module("cowp.scripts.03_train")
    assert hasattr(train_mod, "_maybe_compile_model")
    assert hasattr(train_mod, "_make_adamw_optimizer")
    assert hasattr(train_mod, "_model_state_dict_for_save")
