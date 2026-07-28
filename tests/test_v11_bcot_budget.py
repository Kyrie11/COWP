from __future__ import annotations

import copy

import torch

from cowp.models.losses import set_transport_loss
from cowp.models.set_transport_head import SetTransportCertificateHead


def _head_inputs(*, agents: int = 2, burden: float = 0.4):
    torch.manual_seed(41)
    b, k, n, a, m, r, t, d = 1, 3, 5, agents, 4, 8, 12, 16
    natural = {
        "mode_latent": torch.randn(b, a, m, d),
        "logits": torch.randn(b, a, m),
        "source_logits": torch.randn(b, a, m, 4),
        "priority_logits": torch.full((b, a, m), 2.0),
        "traj": torch.randn(b, a, m, t, 7),
    }
    response = {
        "safe_logits": torch.full((b, k, a, r), 4.0),
        "low_logits": torch.full((b, k, a, r), 4.0),
        "valid_logits": torch.full((b, k, a, r), 4.0),
        "mode_logits": torch.zeros(b, k, a, r),
        "root_logits": torch.randn(b, k, a, r, m),
        "burden_total": torch.full((b, k, a, r), burden),
    }
    return {
        "z_agent": torch.randn(b, n, d),
        "z_candidate": torch.randn(b, k, d),
        "z_graph": torch.randn(b, d),
        "critical_indices": torch.arange(a).view(1, a),
        "critical_mask": torch.ones(b, a, dtype=torch.bool),
        "natural": natural,
        "response": response,
        "beta": torch.full((b, a), 0.65),
        "candidate_traj": torch.randn(b, k, t, 7),
        "natural_traj": natural["traj"],
        "calibration_scale": 0.0,
    }


def test_bcot_candidate_risk_is_bounded_and_monotone_in_burden():
    head = SetTransportCertificateHead(d_model=16, hidden=24, geometry_steps=8, response_topk=8)
    low = head(**_head_inputs(burden=0.1))
    high = head(**_head_inputs(burden=1.3))
    assert low["candidate_transport_risk"].shape == (1, 3)
    assert torch.all((low["candidate_transport_risk"] >= 0.0) & (low["candidate_transport_risk"] <= 1.0))
    assert torch.all(high["candidate_transport_risk"] >= low["candidate_transport_risk"] - 1.0e-6)


def test_bcot_ignores_padded_critical_agents():
    head = SetTransportCertificateHead(d_model=16, hidden=24, geometry_steps=8, response_topk=8)
    base = _head_inputs(agents=2)
    base["critical_mask"][:, 1] = False
    changed = copy.deepcopy(base)
    changed["natural"]["mode_latent"][:, 1].mul_(100.0)
    changed["natural"]["logits"][:, 1].add_(50.0)
    changed["natural"]["traj"][:, 1].add_(1000.0)
    changed["natural_traj"] = changed["natural"]["traj"]
    changed["response"]["safe_logits"][:, :, 1].fill_(-20.0)
    changed["response"]["low_logits"][:, :, 1].fill_(-20.0)
    changed["response"]["burden_total"][:, :, 1].fill_(10.0)
    out0 = head(**base)
    out1 = head(**changed)
    assert torch.allclose(out0["candidate_transport_risk"], out1["candidate_transport_risk"], atol=1.0e-6)
    assert torch.allclose(out0["candidate_transport_uncertainty"], out1["candidate_transport_uncertainty"], atol=1.0e-6)


def test_root_existential_does_not_inflate_uniform_duplicate_slots():
    head = SetTransportCertificateHead(d_model=16, hidden=24, geometry_steps=8, response_topk=8)
    x = _head_inputs(agents=1)
    x["response"]["root_logits"].zero_()
    out = head(**x)
    # A fuzzy existential max gives at most one uniform-root contribution (1/M),
    # rather than accumulating duplicate slots toward one.
    assert float(out["root_response_exist"].max()) <= 0.26


def _minimal_loss_case(risk_values: torch.Tensor):
    b, k, a = 1, 2, 1
    pred = {
        "exist_logits": torch.zeros(b, k, a),
        "opr": torch.ones(b, k, a),
        "min_safe_burden": torch.zeros(b, k, a),
        "natural_conflict_mass": torch.zeros(b, k, a),
        "response_exist_low_safe": torch.full((b, k, a), 0.5),
        "candidate_transport_risk": risk_values.view(b, k),
    }
    batch = {
        "cowp/candidates/valid": torch.ones(b, k, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(b, a, dtype=torch.bool),
        "cowp/witness/exists": torch.zeros(b, k, a),
        "cowp/witness/opr": torch.ones(b, k, a),
        "cowp/witness/burden_total": torch.zeros(b, k, a),
        "cowp/witness/natural_conflict_mass": torch.zeros(b, k, a),
        # Candidate 0 is NCF; candidate 1 is false-safe.
        "cowp/candidates/noncoercive_feasible": torch.tensor([[1, 0]], dtype=torch.bool),
        "cowp/candidates/false_safe": torch.tensor([[0, 1]], dtype=torch.bool),
    }
    return pred, batch


def test_candidate_budget_loss_rewards_ncf_false_safe_ordering():
    good_pred, batch = _minimal_loss_case(torch.tensor([0.1, 0.9], requires_grad=True))
    bad_pred, _ = _minimal_loss_case(torch.tensor([0.9, 0.1], requires_grad=True))
    weights = {
        "set_transport_candidate_budget": 1.0,
        "set_transport_candidate_margin": 0.1,
    }
    good_pred["candidate_transport_risk"].retain_grad()
    good = set_transport_loss(good_pred, batch, weights)
    bad = set_transport_loss(bad_pred, batch, weights)
    assert torch.isfinite(good["candidate_budget"])
    assert good["candidate_budget"] < bad["candidate_budget"]
    good["loss"].backward()
    assert good_pred["candidate_transport_risk"].grad is not None
