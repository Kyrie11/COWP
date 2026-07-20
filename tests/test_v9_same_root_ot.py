from __future__ import annotations

import torch

from cowp.models.losses import _sinkhorn_transport_plan
from cowp.models.set_transport_head import SetTransportCertificateHead


def test_sinkhorn_plan_preserves_both_set_marginals():
    cost = torch.tensor([[[[0.0, 3.0], [3.0, 0.0]]]], dtype=torch.float32)
    a = torch.tensor([[[0.6, 0.4]]], dtype=torch.float32)
    b = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
    valid = torch.tensor([[[True, True]]])
    plan = _sinkhorn_transport_plan(cost, a, b, valid, epsilon=0.5, iterations=80)
    assert torch.allclose(plan.sum(dim=-1), a, atol=2e-3)
    assert torch.allclose(plan.sum(dim=-2), b, atol=2e-3)
    diagonal = plan[0, 0, 0, 0] + plan[0, 0, 1, 1]
    off_diagonal = plan[0, 0, 0, 1] + plan[0, 0, 1, 0]
    assert diagonal > off_diagonal


def test_set_transport_exposes_intervention_burden_field():
    torch.manual_seed(3)
    b, k, a, m, r, d = 1, 2, 1, 3, 4, 8
    head = SetTransportCertificateHead(d_model=d, hidden=d)
    out = head(
        z_agent=torch.randn(b, 2, d),
        z_candidate=torch.randn(b, k, d),
        z_graph=torch.randn(b, d),
        critical_indices=torch.zeros(b, a, dtype=torch.long),
        natural={
            "mode_latent": torch.randn(b, a, m, d),
            "logits": torch.zeros(b, a, m),
            "source_logits": torch.zeros(b, a, m, 4),
            "priority_logits": torch.zeros(b, a, m),
            "valid_logits": torch.full((b, a, m), 5.0),
            "low_neutral_logits": torch.full((b, a, m), 5.0),
            "neutral_burden": torch.zeros(b, a, m),
        },
        response={
            "safe_logits": torch.full((b, k, a, r), 4.0),
            "low_logits": torch.full((b, k, a, r), 4.0),
            "valid_logits": torch.full((b, k, a, r), 4.0),
            "mode_logits": torch.zeros(b, k, a, r),
            "burden_total": torch.full((b, k, a, r), 0.2),
        },
        beta=torch.full((b, a), 0.65),
        calibration_scale=0.0,
    )
    assert out["mode_burden_under"].shape == (b, k, a, m)
    assert torch.all(out["mode_burden_under"] >= 0.0)
    expected = (1.0 - out["mode_conflict_prob"]) * out["mode_low_burden_prob"]
    assert torch.allclose(out["mode_low_safe_prob"], expected, atol=1e-6)
