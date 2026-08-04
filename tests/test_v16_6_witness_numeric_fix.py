from __future__ import annotations

import importlib

import torch

from cowp.models.candidate_encoder import CandidateEncoder
from cowp.models.coordinate import ego_centric_inputs


def _agent_history_with_large_global_origin() -> torch.Tensor:
    # [B,N,H,11] uses x,y,z,length,width,height,heading,vx,vy,speed,valid.
    hist = torch.zeros(1, 1, 2, 11)
    hist[..., 0] = 1_000_000.0
    hist[..., 1] = -2_000_000.0
    hist[..., 3] = 4.8
    hist[..., 4] = 1.9
    hist[..., 6] = 0.35
    hist[..., 10] = 1.0
    return hist


def test_ego_centric_transform_preserves_invalid_candidate_padding() -> None:
    hist = _agent_history_with_large_global_origin()
    cand = torch.zeros(1, 2, 4, 7)
    # One physically valid trajectory around the global ego origin.
    cand[0, 0, :, 0] = 1_000_000.0 + torch.arange(4).float()
    cand[0, 0, :, 1] = -2_000_000.0
    cand[0, 0, :, 2] = 0.35
    cand[0, 0, :, 3] = 5.0
    cand[0, 0, :, 5:7] = torch.tensor([4.8, 1.9])
    valid = torch.tensor([[True, False]])

    _, enc_cand, _ = ego_centric_inputs(
        hist,
        cand,
        None,
        torch.tensor([0]),
        candidate_valid=valid,
    )
    assert enc_cand is not None
    assert torch.isfinite(enc_cand).all()
    assert torch.count_nonzero(enc_cand[0, 1]) == 0
    assert enc_cand[0, 0, :, :2].abs().max() < 10.0


def test_candidate_encoder_masks_padding_and_has_finite_backward() -> None:
    torch.manual_seed(7)
    encoder = CandidateEncoder(d_model=16, macro_count=13, dropout=0.0)
    traj = torch.randn(2, 3, 8, 7, requires_grad=True)
    # Reproduce the old invalid-slot failure mode: huge translated padding and
    # non-finite values that must never enter the recurrent kernel.
    with torch.no_grad():
        traj[:, 2, :, :] = 1.0e30
        traj[0, 2, 0, 0] = float("nan")
    valid = torch.tensor([[True, True, False], [True, False, False]])
    macro = torch.tensor([[0, 1, 12], [2, 12, 12]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = encoder(traj, macro, valid_mask=valid)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[~valid]) == 0

    loss = out.square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert traj.grad is not None
    assert torch.count_nonzero(traj.grad[~valid]) == 0



def test_valid_candidate_nonfinite_is_reported_not_silently_repaired() -> None:
    encoder = CandidateEncoder(d_model=16, dropout=0.0)
    traj = torch.zeros(1, 1, 4, 7)
    traj[0, 0, 2, 3] = float("nan")
    try:
        encoder(traj, torch.zeros(1, 1, dtype=torch.long), valid_mask=torch.ones(1, 1, dtype=torch.bool))
    except FloatingPointError as exc:
        assert "[0, 0, 2, 3]" in str(exc)
    else:
        raise AssertionError("valid non-finite candidate input must fail loudly")

def test_candidate_feature_scale_does_not_break_old_checkpoints() -> None:
    encoder = CandidateEncoder(d_model=16)
    assert "candidate_feature_scale" not in encoder.state_dict()


def test_grad_scaler_is_fp16_only() -> None:
    train_mod = importlib.import_module("cowp.scripts.03_train")
    assert train_mod._use_grad_scaler(True, True, torch.float16)
    assert not train_mod._use_grad_scaler(True, True, torch.bfloat16)
    assert not train_mod._use_grad_scaler(False, True, torch.float16)
    assert not train_mod._use_grad_scaler(True, False, torch.float16)

from cowp.models.losses import paper_aligned_supervision_batch
from cowp.label.witness import _weighted_upper_cvar


def test_reused_v9_targets_are_rebuilt_from_root_mass_and_same_root_tail() -> None:
    # Two active natural roots and one padded root.  Root 0 is blocked and needs
    # a high-burden response; root 1 is retained.  A conditional retained ratio
    # would be misleading here, while the paper's OPR is retained probability mass.
    batch = {
        "cowp/candidates/valid": torch.tensor([[True]]),
        "cowp/candidates/conventional_safe": torch.tensor([[True]]),
        "cowp/critical/valid": torch.tensor([[True]]),
        "cowp/natural/valid": torch.tensor([[[True, True, False]]]),
        "cowp/natural/weight": torch.tensor([[[0.9, 0.1, 0.0]]]),
        "cowp/natural/beta": torch.tensor([[0.5]]),
        "cowp/transport/mode_valid": torch.tensor([[[[True, True, False]]]]),
        "cowp/transport/mode_conflict": torch.tensor([[[[True, False, False]]]]),
        "cowp/transport/mode_retained_low_safe": torch.tensor([[[[False, True, False]]]]),
        "cowp/transport/response_root_index": torch.tensor([[[[0, 1]]]]),
        "cowp/response/valid": torch.tensor([[[[True, True]]]]),
        "cowp/response/is_safe": torch.tensor([[[[True, True]]]]),
        "cowp/response/burden_total": torch.tensor([[[[1.2, 0.2]]]]),
        "cowp/response/burden_components": torch.tensor(
            [[[[[0.2, 0.1, 0.2, 0.1, 2.0, 0.1],
                [0.1, 0.1, 0.1, 0.1, 2.0, 0.1]]]]]
        ),
        # Deliberately stale v9 labels.  The adapter must replace decision targets
        # while preserving token/interval supervision only where old explanations exist.
        "cowp/witness/exists": torch.tensor([[[False]]]),
        "cowp/witness/burden_total": torch.tensor([[[1.8]]]),
        "cowp/witness/burden_components": torch.ones(1, 1, 1, 6),
        "cowp/witness/opr": torch.tensor([[[1.0]]]),
        "cowp/witness/c_i": torch.tensor([[[0.0]]]),
    }
    cfg = {
        "paper_aligned_witness_targets": 1.0,
        "set_transport_probability_floor": 0.02,
        "set_transport_cvar_tail_mass": 0.25,
        "witness_conflict_mass_floor": 0.10,
        "witness_burden_gamma": 0.10,
        "witness_opr_alpha": 0.35,
    }
    out = paper_aligned_supervision_batch(batch, cfg)

    # Floor is distributed only over the two valid roots:
    # p0=.98*.9+.02/2=.892, p1=.108, padded root remains zero.
    assert torch.allclose(out["cowp/witness/natural_conflict_mass"], torch.tensor([[[0.892]]]), atol=1e-6)
    assert torch.allclose(out["cowp/witness/opr"], torch.tensor([[[0.108]]]), atol=1e-6)
    # Conflicted root 0 has [1.2-.5]+=.7, so its conflict-conditioned CVaR is .7.
    assert torch.allclose(out["cowp/witness/tail_burden_excess"], torch.tensor([[[0.7]]]), atol=1e-6)
    assert bool(out["cowp/witness/exists"].item())
    assert bool(out["cowp/candidates/false_safe"].item())
    assert not bool(out["cowp/candidates/noncoercive_feasible"].item())
    # Primitive burden remains independent of OPR; option component is removed.
    assert torch.allclose(out["cowp/witness/burden_total"], torch.tensor([[[0.2]]]))
    assert out["cowp/witness/burden_components"][..., 4].abs().sum() == 0
    assert not bool(out["cowp/witness/explanation_valid"].item())


def test_weighted_upper_cvar_uses_probability_mass_not_root_count() -> None:
    # Worst value carries only 10% probability; at rho=.5 the remaining 40% tail
    # mass must come from the second root: (0.1*1 + 0.4*0.5)/0.5 = 0.6.
    value = _weighted_upper_cvar(
        values=torch.tensor([1.0, 0.5, 0.0]).numpy(),
        weights=torch.tensor([0.1, 0.8, 0.1]).numpy(),
        tail_mass=0.5,
    )
    assert abs(value - 0.6) < 1e-7

from cowp.models.set_transport_head import SetTransportCertificateHead


def _tiny_set_transport_inputs() -> dict:
    b, k, a, m, r, d = 1, 2, 1, 3, 4, 8
    return {
        "z_agent": torch.randn(b, 2, d),
        "z_candidate": torch.randn(b, k, d),
        "z_graph": torch.randn(b, d),
        "critical_indices": torch.zeros(b, a, dtype=torch.long),
        "natural": {
            "mode_latent": torch.randn(b, a, m, d),
            "logits": torch.zeros(b, a, m),
            "source_logits": torch.zeros(b, a, m, 4),
            "priority_logits": torch.zeros(b, a, m),
        },
        "response": {
            "safe_logits": torch.zeros(b, k, a, r),
            "low_logits": torch.zeros(b, k, a, r),
            "valid_logits": torch.zeros(b, k, a, r),
            "mode_logits": torch.zeros(b, k, a, r),
            "root_logits": torch.zeros(b, k, a, r, m),
            "burden_total": torch.zeros(b, k, a, r),
        },
        "beta": torch.full((b, a), 0.65),
    }


def test_bfloat16_retain_probability_to_logit_stays_finite_at_saturation() -> None:
    head = SetTransportCertificateHead(d_model=8, hidden=8)
    # Force P(conflict)≈0 and P(retain|no-conflict)≈1.  In the old code the
    # BF16 clamp upper bound 1-1e-5 rounded back to exactly 1 and logit became Inf.
    with torch.no_grad():
        final = head.mode_out[-1]
        final.weight.zero_()
        final.bias.copy_(torch.tensor([-100.0, 100.0, 0.0, 0.0, 0.0]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = head(**_tiny_set_transport_inputs(), calibration_scale=0.0)
    assert out["mode_retain_prob"].dtype == torch.float32
    assert torch.isfinite(out["mode_retain_logits"]).all()
    assert float(out["mode_retain_prob"].detach().max()) < 1.0
    assert float(out["mode_retain_logits"].detach().max()) > 10.0
    out["mode_retain_logits"].sum().backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_batched_nonfinite_output_scan_reports_alias_once() -> None:
    import importlib

    train_mod = importlib.import_module("cowp.scripts.03_train")
    x = torch.tensor([1.0, float("inf")])
    pred = {"a": x, "nested": {"same_alias": x, "finite": torch.ones(3)}}
    assert train_mod._nonfinite_tensor_paths(pred) == ["pred.a"]
