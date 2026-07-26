from __future__ import annotations

import json
import subprocess
import sys

import torch

from cowp.core.constants import NaturalSource
from cowp.models.losses import natural_loss
from cowp.models.natural_decoder import NaturalDecoder
from cowp.waymax_eval.rollout import _candidate_certificate_scores


def test_cnob_zero_initialization_is_exact_analytic_basis():
    decoder = NaturalDecoder(d_model=16, modes=24, future_steps=8, decoder_type="typed_causal_dynamics")
    z = torch.randn(2, 5, 16)
    idx = torch.tensor([[1, 3], [0, 4]])
    anchor = torch.tensor([
        [[0.0, 0.0, 0.2, 4.0, 0.8, 4.5, 1.8], [1.0, 2.0, -0.1, 2.0, -0.2, 4.0, 1.7]],
        [[0.0, 1.0, 0.4, 3.0, 1.2, 4.5, 1.8], [2.0, 0.0, -0.3, 1.5, -0.4, 4.0, 1.7]],
    ])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    assert decoder.uses_typed_basis and decoder.uses_dynamic_residual
    assert torch.allclose(out["traj"], out["base_traj"], atol=1e-7)
    assert torch.count_nonzero(out["residual"]) == 0
    assert torch.count_nonzero(out["controls"]) == 0


def test_cnob_control_residual_is_kinematically_consistent_and_size_constant():
    decoder = NaturalDecoder(d_model=8, modes=24, future_steps=12, decoder_type="cnob_dynamics")
    with torch.no_grad():
        decoder.temporal_head[-1].bias.copy_(torch.tensor([0.4, 0.2, 0.1, 0.15, -0.1, 0.05, 0.0]))
    z = torch.randn(1, 2, 8)
    idx = torch.tensor([[0]])
    anchor = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 0.0, 4.5, 1.8]]])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    absolute = out["traj"] + anchor[:, :, None, None, :]
    fd = (absolute[..., 1:, 0:2] - absolute[..., :-1, 0:2]) / 0.1
    vm = 0.5 * (absolute[..., 1:, 3:5] + absolute[..., :-1, 3:5])
    assert torch.max(torch.linalg.norm(fd - vm, dim=-1)).item() < 2e-4
    assert torch.count_nonzero(out["residual"][..., 5:7]) == 0


def test_new_natural_loss_rewards_real_obs_gain():
    base = torch.zeros(1, 1, 3, 2, 7)
    base[:, :, 0, :, 0] = 1.0
    learned = base.clone()
    learned[:, :, 0, :, 0] = 0.0
    pred = {
        "traj": learned,
        "base_traj": base,
        "residual": learned - base,
        "controls": torch.zeros(1, 1, 3, 2, 5),
        "logits": torch.zeros(1, 1, 3),
        "source_logits": torch.zeros(1, 1, 3, 4),
        "priority_logits": torch.zeros(1, 1, 3),
        "mode_source": torch.tensor([0, 1, 2]),
    }
    batch = {
        "cowp/natural/traj": torch.zeros(1, 1, 1, 2, 7),
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(1, 1, 1),
        "cowp/natural/source": torch.zeros(1, 1, 1),
        "cowp/natural/priority_preserved": torch.zeros(1, 1, 1),
        "cowp/critical/valid": torch.ones(1, 1, dtype=torch.bool),
    }
    loss = natural_loss(pred, batch, {"natural_dt": 0.1, "natural_obs_gain_margin_m": 0.05})
    assert loss["residual_obs_gain"].item() > 0.9
    assert loss["residual_obs_shortfall"].item() < 1e-6


def test_certificate_fallback_is_disabled_when_protocol_says_so():
    scores = torch.tensor([[0.0, 2.0, 4.0]])
    pred = {
        "candidate_ncf_logit": torch.zeros_like(scores),
        "candidate_false_safe_logit": torch.zeros_like(scores),
        "candidate_quality_logit": torch.zeros_like(scores),
        "outcome": {"collision_logit": torch.tensor([[10.0, -10.0, 10.0]])},
    }
    cfg = {"planning": {
        "candidate_cert_allow_hybrid_fallback": False,
        "candidate_cert_hybrid_fallback_mix": 1.0,
        "candidate_cert_flat_fallback_mix": 1.0,
    }}
    ncf, fs, q = _candidate_certificate_scores(pred, scores, cfg, torch.ones_like(scores, dtype=torch.bool))
    assert torch.allclose(ncf, torch.full_like(scores, 0.5))
    assert torch.allclose(fs, torch.full_like(scores, 0.5))
    assert torch.allclose(q, torch.full_like(scores, 0.5))


def test_effectiveness_gate_rejects_inert_residual(tmp_path):
    source = {
        "distributions": {
            "all/8s/learned": {"mean": 1.8}, "all/8s/gain": {"mean": 0.0},
            "source_0/8s/learned": {"mean": 3.0}, "source_0/8s/gain": {"mean": 0.0},
            "source_1/8s/gain": {"mean": 0.0}, "source_2/8s/gain": {"mean": 0.0},
            "kinematic/velocity_error_mps": {"mean": 0.01},
            "kinematic/yaw_error_rad": {"mean": 0.01},
            "residual/endpoint_m": {"p99": 0.0},
            "residual/soft_path_ratio": {"weighted_mean": 0.0},
            "residual/soft_path_violation": {"weighted_mean": 0.0},
            "residual/projected_emergency_path_ratio": {"p99": 0.0},
        },
        "mode_usage": {f"source_{i}": {"effective_modes": 3.0} for i in range(3)},
    }
    inp, out = tmp_path / "diag.json", tmp_path / "gate.json"
    inp.write_text(json.dumps(source))
    proc = subprocess.run([
        sys.executable, "-m", "cowp.scripts.40_gate_natural_effectiveness",
        "--report", str(inp), "--output", str(out),
    ], check=False)
    assert proc.returncode == 2
    assert json.loads(out.read_text())["checks"]["residual_improves_obs"] is False


def test_obs_capacity_scale_zero_matches_neutral_control_limits() -> None:
    decoder = NaturalDecoder(
        d_model=16, modes=24, future_steps=8,
        decoder_type="typed_causal_dynamics", obs_capacity_scale=0.0,
    )
    obs = decoder.mode_source == int(NaturalSource.OBS)
    neu = decoder.mode_source == int(NaturalSource.NEU)
    for name in (
        "control_accel_long_scale", "control_accel_lat_scale",
        "control_jerk_long_scale", "control_jerk_lat_scale", "control_yaw_rate_scale",
        "typed_residual_gate_bias",
    ):
        values = getattr(decoder, name)
        assert torch.allclose(values[obs].mean(), values[neu].mean())


def test_natural_ablation_comparison_attributes_components() -> None:
    import importlib

    module = importlib.import_module("cowp.scripts.41_compare_natural_ablations")

    def report(
        all_8s: float, obs: float, soft_ratio: float, violation: float,
        neu: float = 1.0, prio: float = 1.1,
    ) -> dict:
        dist = {}
        for key, value in {
            "all/8s/learned": all_8s, "source_0/8s/learned": obs,
            "source_1/8s/learned": neu, "source_2/8s/learned": prio,
            "all/8s/gain": 0.1, "source_0/8s/gain": 0.2,
            "residual/soft_path_ratio": soft_ratio,
            "residual/soft_path_violation": violation,
            "residual/projection_active": 0.01,
        }.items():
            dist[key] = {"weighted_mean": value}
        dist["residual/projected_emergency_path_ratio"] = {"p99": 1.0}
        return {"distributions": dist}

    result = module.compare_reports(
        report(2.0, 3.0, 0.80, 0.10),
        report(2.05, 3.1, 0.82, 0.11),
        report(2.03, 3.05, 0.90, 0.16),
        min_capacity_obs_gain_m=0.05,
        min_mass_ratio_reduction=0.03,
        min_mass_violation_reduction=0.02,
    )
    assert result["pass"] is True
    assert result["checks"]["obs_capacity_improves_obs"] is True
    assert result["checks"]["mass_envelope_reduces_probability_weighted_ratio"] is True

def test_mode_usage_loss_supports_realistic_agent_mode_root_dimensions() -> None:
    """Regression for the v16 A=6/M=24 broadcast failure seen before epoch -1."""
    torch.manual_seed(7)
    B, A, M, R, T = 2, 6, 24, 24, 8
    decoder = NaturalDecoder(
        d_model=16, modes=M, future_steps=T, decoder_type="typed_causal_dynamics"
    )
    z = torch.randn(B, A + 1, 16)
    critical_indices = torch.arange(A).repeat(B, 1)
    anchor = torch.zeros(B, A, 7)
    anchor[..., 3] = 5.0
    pred = decoder(z, critical_indices, anchor7=anchor, dt=0.1)

    source = torch.tensor([0] * 8 + [1] * 8 + [2] * 8).view(1, 1, R).expand(B, A, -1)
    gt = pred["base_traj"].detach().clone()  # [B,A,M,T,7]
    batch = {
        "cowp/natural/traj": gt,
        "cowp/natural/valid": torch.ones(B, A, R, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(B, A, R),
        "cowp/natural/source": source,
        "cowp/natural/priority_preserved": (source == int(NaturalSource.PRIO)).float(),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
    }
    losses = natural_loss(pred, batch, {"natural_dt": 0.1, "natural_mode_usage": 0.03})
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["mode_usage"])
    losses["loss"].backward()


def test_mode_usage_loss_handles_batch_varying_expanded_mode_source() -> None:
    B, A, M, R, T = 2, 3, 6, 6, 3
    mode_source = torch.tensor([0, 0, 1, 1, 2, 2]).view(1, 1, M).expand(B, A, -1).clone()
    pred = {
        "traj": torch.zeros(B, A, M, T, 7, requires_grad=True),
        "base_traj": torch.zeros(B, A, M, T, 7),
        "residual": torch.zeros(B, A, M, T, 7),
        "controls": torch.zeros(B, A, M, T, 5),
        "logits": torch.zeros(B, A, M, requires_grad=True),
        "source_logits": torch.zeros(B, A, M, 4),
        "priority_logits": torch.zeros(B, A, M),
        "mode_source": mode_source,
    }
    source = torch.tensor([0, 0, 1, 1, 2, 2]).view(1, 1, R).expand(B, A, -1)
    batch = {
        "cowp/natural/traj": torch.zeros(B, A, R, T, 7),
        "cowp/natural/valid": torch.ones(B, A, R, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(B, A, R),
        "cowp/natural/source": source,
        "cowp/natural/priority_preserved": (source == 2).float(),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
    }
    losses = natural_loss(pred, batch, {"natural_dt": 0.1})
    assert torch.isfinite(losses["mode_usage"])


def test_cnob_stopped_agent_yaw_uses_absolute_velocity_heading() -> None:
    """Regression for the v16.3 stopped-root heading double-counting bug."""
    decoder = NaturalDecoder(
        d_model=8,
        modes=24,
        future_steps=12,
        decoder_type="typed_causal_dynamics",
    )
    with torch.no_grad():
        # Produce a forward/lateral acceleration from a stopped, rotated anchor.
        decoder.temporal_head[-1].bias.copy_(
            torch.tensor([0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
    z = torch.randn(1, 1, 8)
    idx = torch.tensor([[0]])
    anchor = torch.tensor([[[0.0, 0.0, 0.7, 0.0, 0.0, 4.5, 1.8]]])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    absolute = out["traj"] + anchor[:, :, None, None, :]
    speed = torch.linalg.norm(absolute[..., 3:5], dim=-1)
    moving = speed > 0.5
    assert bool(moving.any())
    vel_yaw = torch.atan2(absolute[..., 4], absolute[..., 3])
    yaw_err = torch.abs(torch.atan2(
        torch.sin(absolute[..., 2] - vel_yaw),
        torch.cos(absolute[..., 2] - vel_yaw),
    ))
    assert torch.max(yaw_err[moving]).item() < 2.0e-5


def test_cnob_integrated_residual_respects_source_endpoint_budgets_and_gradients() -> None:
    decoder = NaturalDecoder(
        d_model=8,
        modes=24,
        future_steps=80,
        decoder_type="typed_causal_dynamics",
        residual_endpoint_budget_obs_m=20.0,
        residual_endpoint_budget_neu_m=8.0,
        residual_endpoint_budget_prio_m=6.0,
    )
    with torch.no_grad():
        decoder.temporal_head[-1].bias.copy_(
            torch.tensor([8.0, 8.0, 0.5, 8.0, 8.0, 0.5, 0.0])
        )
    z = torch.randn(2, 2, 8, requires_grad=True)
    idx = torch.tensor([[0], [1]])
    anchor = torch.tensor([
        [[0.0, 0.0, 0.2, 3.0, 0.5, 4.5, 1.8]],
        [[1.0, 2.0, -0.4, 1.0, -0.2, 4.0, 1.7]],
    ])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    endpoint = torch.linalg.norm(out["residual"][..., -1, 0:2], dim=-1)
    emergency_budget = out["residual_emergency_budget_m"][None, None, :]
    assert torch.all(endpoint <= emergency_budget + 2.0e-4)
    assert torch.max(out["projected_residual_emergency_path_ratio"]).item() <= 1.0 + 2.0e-4
    loss = out["traj"].square().mean() + out["controls"].square().mean()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_cnob_soft_trust_region_has_radial_gradient_before_projection() -> None:
    from cowp.models.losses import _natural_residual_trust_region_losses

    decoder = NaturalDecoder(
        d_model=8,
        modes=24,
        future_steps=80,
        decoder_type="typed_causal_dynamics",
        residual_endpoint_budget_obs_m=0.5,
        residual_endpoint_budget_neu_m=0.5,
        residual_endpoint_budget_prio_m=0.5,
    )
    with torch.no_grad():
        decoder.temporal_head[-1].bias.copy_(
            torch.tensor([2.0, 1.5, 0.2, 1.0, 0.8, 0.2, 0.0])
        )
    z = torch.randn(1, 1, 8, requires_grad=True)
    idx = torch.tensor([[0]])
    anchor = torch.tensor([[[0.0, 0.0, 0.3, 2.0, 0.5, 4.5, 1.8]]])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    assert torch.max(out["raw_residual_endpoint_m"]).item() > 0.5
    penalty, mean_ratio, saturation = _natural_residual_trust_region_losses(
        out, torch.ones(1, 1, dtype=torch.bool), soft_ratio=0.75
    )
    assert penalty.item() > 0.0
    assert mean_ratio.item() > 1.0
    assert saturation.item() > 0.0
    penalty.backward()
    bias_grad = decoder.temporal_head[-1].bias.grad
    assert bias_grad is not None and torch.isfinite(bias_grad).all()
    assert bias_grad.abs().sum().item() > 0.0
