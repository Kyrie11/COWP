from __future__ import annotations

import torch

from cowp.models.response_decoder import ResponseDecoder
from cowp.models.set_transport_head import SetTransportCertificateHead


def _inputs(*, burden: float):
    torch.manual_seed(7)
    b, k, a, m, r, d = 1, 2, 1, 3, 4, 8
    natural = {
        "mode_latent": torch.randn(b, a, m, d),
        "logits": torch.zeros(b, a, m),
        "source_logits": torch.zeros(b, a, m, 4),
        "priority_logits": torch.full((b, a, m), 2.0),
    }
    response = {
        "safe_logits": torch.full((b, k, a, r), 5.0),
        "low_logits": torch.full((b, k, a, r), 5.0),
        "valid_logits": torch.full((b, k, a, r), 5.0),
        "mode_logits": torch.zeros(b, k, a, r),
        "burden_total": torch.full((b, k, a, r), burden),
    }
    return {
        "z_agent": torch.randn(b, 2, d),
        "z_candidate": torch.randn(b, k, d),
        "z_graph": torch.randn(b, d),
        "critical_indices": torch.zeros(b, a, dtype=torch.long),
        "natural": natural,
        "response": response,
        "beta": torch.full((b, a), 0.65),
    }


def test_compact_response_decode_does_not_allocate_trajectory():
    decoder = ResponseDecoder(d_model=8, responses=4, future_steps=6)
    out = decoder(
        torch.randn(1, 2, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 8),
        torch.zeros(1, 1, dtype=torch.long),
        decode_traj=False,
    )
    assert "traj" not in out
    assert out["safe_logits"].shape == (1, 3, 1, 4)
    assert out["source_logits"].shape == (1, 3, 1, 4, 4)


def test_set_certificate_is_monotone_in_required_safe_burden():
    head = SetTransportCertificateHead(d_model=8, hidden=8)
    # Isolate the analytic certificate from its bounded learned calibration.
    low = head(**_inputs(burden=0.20), calibration_scale=0.0, alpha_opr=0.10)
    high = head(**_inputs(burden=1.20), calibration_scale=0.0, alpha_opr=0.10)
    assert low["witness_prob"].shape == (1, 2, 1)
    assert torch.all(high["min_safe_burden"] > low["min_safe_burden"])
    assert torch.all(high["witness_prob"] > low["witness_prob"])
    assert torch.all((low["opr"] >= 0.0) & (low["opr"] <= 1.0))
