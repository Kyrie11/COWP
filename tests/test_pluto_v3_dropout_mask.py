from __future__ import annotations

import torch

from cowp.external_baselines.pluto_cowp import COWPPLUTO, pluto_loss


def test_pluto_contrastive_agent_dropout_updates_valid_mask(monkeypatch):
    torch.manual_seed(4)
    B, N, T, F = 2, 3, 11, 9
    model = COWPPLUTO(
        future_len=8, d_model=16, num_heads=4, encoder_layers=1,
        lateral_queries=2, longitudinal_queries=2, dropout=0.0,
    ).eval()
    inputs = {
        "agents": torch.randn(B, N, T, F),
        "agent_valid": torch.ones(B, N, T, dtype=torch.bool),
        "map_lanes": torch.randn(B, 2, 10, 7),
        "route": torch.cat([torch.randn(B, 8, 3), torch.ones(B, 8, 1)], dim=-1),
    }
    seen_valid = []
    original_encode = model.encode_scene

    def wrapped_encode(inp):
        seen_valid.append(inp["agent_valid"].detach().clone())
        return original_encode(inp)

    model.encode_scene = wrapped_encode  # type: ignore[method-assign]

    def all_drop_rand(size, *args, **kwargs):
        # keep = rand > 0.15 => false for every actor except ego, which loss
        # explicitly forces to true.
        return torch.zeros(size, device=kwargs.get("device", None))

    monkeypatch.setattr(torch, "rand", all_drop_rand)
    gt = torch.randn(B, 8, 2)
    gt_valid = torch.ones(B, 8, dtype=torch.bool)
    loss, _ = pluto_loss(model, inputs, gt, gt_valid, contrast_weight=0.05)
    assert torch.isfinite(loss)
    assert len(seen_valid) >= 2
    aug_valid = seen_valid[-1]
    assert bool(aug_valid[:, 0].all())
    assert not bool(aug_valid[:, 1:].any())
