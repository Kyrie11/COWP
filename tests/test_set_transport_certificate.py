from __future__ import annotations

import pytest
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


def test_generic_response_burden_is_auxiliary_to_root_conditioned_certificate():
    head = SetTransportCertificateHead(d_model=8, hidden=8)
    # The generic response bank still reconstructs an interpretable global
    # minimum, but it must not override the RCOT root-conditioned certificate.
    low = head(**_inputs(burden=0.20), calibration_scale=0.0, alpha_opr=0.10)
    high = head(**_inputs(burden=1.20), calibration_scale=0.0, alpha_opr=0.10)
    assert low["witness_prob"].shape == (1, 2, 1)
    assert torch.all(high["min_safe_burden"] > low["min_safe_burden"])
    assert torch.allclose(high["witness_prob"], low["witness_prob"])
    assert torch.all((low["opr"] >= 0.0) & (low["opr"] <= 1.0))


def test_root_conditioned_burden_is_monotone_in_certificate():
    head = SetTransportCertificateHead(d_model=8, hidden=8)
    inputs = _inputs(burden=0.20)
    with torch.no_grad():
        final = head.mode_out[-1]
        final.weight.zero_()
        final.bias.zero_()
        final.bias[0] = 8.0   # conflict
        final.bias[1] = -8.0  # no retained mass
        final.bias[3] = -8.0  # no low-burden recovery
        final.bias[4] = -8.0  # low b*
    low = head(**inputs, calibration_scale=0.0, alpha_opr=0.10)
    with torch.no_grad():
        head.mode_out[-1].bias[4] = 8.0  # high b*
    high = head(**inputs, calibration_scale=0.0, alpha_opr=0.10)
    assert torch.all(high["root_min_safe_burden"] > low["root_min_safe_burden"])
    assert torch.all(high["tail_burden_excess"] > low["tail_burden_excess"])
    assert torch.all(high["witness_prob"] >= low["witness_prob"])


def test_root_probability_measure_applies_pmin_then_floor_on_active_support():
    head = SetTransportCertificateHead(d_model=8, hidden=8)
    inputs = _inputs(burden=0.20)
    # Replace the three uniform roots with masses 0.95, 0.04, 0.01.  p_min=.03
    # removes only the third root; epsilon=.02 is then distributed over two roots.
    probs = torch.tensor([0.95, 0.04, 0.01])
    inputs["natural"]["logits"] = probs.log().view(1, 1, 3)
    out = head(
        **inputs,
        calibration_scale=0.0,
        root_min_alt_weight=0.03,
        root_probability_floor=0.02,
    )
    base = probs[:2] / probs[:2].sum()
    expected = torch.tensor([
        0.98 * base[0] + 0.01,
        0.98 * base[1] + 0.01,
        0.0,
    ])
    got = out["canonical_root_weight"][0, 0, 0]
    assert torch.allclose(got, expected, atol=1.0e-6)
    assert got.sum().item() == pytest.approx(1.0, abs=1.0e-6)
