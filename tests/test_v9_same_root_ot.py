from __future__ import annotations

import torch

from cowp.models.set_transport_head import SetTransportCertificateHead


def _head_output(*, root_logits: torch.Tensor | None = None):
    torch.manual_seed(3)
    b, k, a, m, r, d, t = 1, 2, 1, 3, 4, 8, 10
    head = SetTransportCertificateHead(d_model=d, hidden=d, geometry_steps=5)
    response = {
        "safe_logits": torch.full((b, k, a, r), 4.0),
        "low_logits": torch.full((b, k, a, r), 4.0),
        "valid_logits": torch.full((b, k, a, r), 4.0),
        "mode_logits": torch.zeros(b, k, a, r),
        "burden_total": torch.full((b, k, a, r), 0.2),
    }
    if root_logits is not None:
        response["root_logits"] = root_logits
    return head(
        z_agent=torch.randn(b, 2, d),
        z_candidate=torch.randn(b, k, d),
        z_graph=torch.randn(b, d),
        critical_indices=torch.zeros(b, a, dtype=torch.long),
        natural={
            "mode_latent": torch.randn(b, a, m, d),
            "logits": torch.zeros(b, a, m),
            "source_logits": torch.zeros(b, a, m, 4),
            "priority_logits": torch.zeros(b, a, m),
            "traj": torch.randn(b, a, m, t, 7),
        },
        response=response,
        beta=torch.full((b, a), 0.65),
        candidate_traj=torch.randn(b, k, t, 7),
        natural_traj=torch.randn(b, a, m, t, 7),
        calibration_scale=0.0,
    )


def test_set_transport_exposes_direct_mode_logits():
    out = _head_output(root_logits=torch.zeros(1, 2, 1, 4, 3))
    assert out["mode_conflict_logits"].shape == (1, 2, 1, 3)
    assert out["mode_retain_logits"].shape == (1, 2, 1, 3)
    assert out["geometry_min_distance_norm"].shape == (1, 2, 1, 3)


def test_uniform_root_assignment_respects_response_mass():
    out = _head_output(root_logits=torch.zeros(1, 2, 1, 4, 3))
    # Four response slots do not each count as an independent full-probability
    # recovery event; response mixture weights sum to one before root aggregation.
    assert float(out["root_response_exist"].max()) < 0.45
