from __future__ import annotations

import torch

from cowp.models.cowp_model import COWPModel
from cowp.models.losses import _pairwise_ade, _pairwise_ade_horizons


def _tiny_cfg() -> dict:
    return {
        "model": {
            "d_state": 11,
            "d_model": 16,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.5,
            "max_agents": 4,
            "max_candidates": 2,
            "max_critical_agents": 2,
            "max_natural_alternatives": 6,
            "max_safe_responses": 4,
            "future_steps": 8,
            "token_count": 7,
            "natural_decoder_type": "typed_causal_dynamics",
            "allow_label_only_state_fallback": False,
        },
        "ablation": {},
    }


def test_pairwise_horizon_reuse_is_exact() -> None:
    torch.manual_seed(5)
    pred = torch.randn(2, 3, 4, 8, 7)
    gt = torch.randn(2, 3, 5, 8, 7)
    cached = _pairwise_ade_horizons(pred, gt, (2, 5, 8))
    for h in (2, 5, 8):
        expected = _pairwise_ade(pred[..., :h, :], gt[..., :h, :])
        assert torch.allclose(cached[h], expected, atol=1e-6, rtol=1e-6)


def test_frozen_graph_encoding_is_deterministic_and_has_no_grad_graph() -> None:
    model = COWPModel(_tiny_cfg()).train()
    for p in model.graph.parameters():
        p.requires_grad_(False)
    history = torch.randn(2, 4, 3, 11)
    history[..., 10] = 1.0
    mask = torch.ones(2, 4, dtype=torch.bool)
    ego_mask = torch.zeros_like(mask)
    ego_mask[:, 0] = True
    a = model._encode_graph(history, mask, None, None, None, None, ego_mask=ego_mask)
    b = model._encode_graph(history, mask, None, None, None, None, ego_mask=ego_mask)
    assert model.graph.training is False
    assert torch.equal(a["z_agent"], b["z_agent"])
    assert a["z_agent"].requires_grad is False
