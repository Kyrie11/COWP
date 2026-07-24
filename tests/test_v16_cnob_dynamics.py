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

    def report(all_8s: float, obs: float, neu: float = 1.0, prio: float = 1.1) -> dict:
        dist = {}
        for key, value in {
            "all/8s/learned": all_8s, "source_0/8s/learned": obs,
            "source_1/8s/learned": neu, "source_2/8s/learned": prio,
            "all/8s/gain": 0.1, "source_0/8s/gain": 0.2,
            "kinematic/velocity_error_mps": 0.02, "kinematic/yaw_error_rad": 0.01,
        }.items():
            dist[key] = {"mean": value}
        return {"distributions": dist}

    result = module.compare_reports(
        report(2.0, 3.0), report(2.1, 3.1), report(2.05, 3.1),
        min_loss_obs_gain_m=0.05, min_loss_overall_gain_m=0.02,
        min_capacity_obs_gain_m=0.05, max_prior_regression_m=0.15,
    )
    assert result["pass"] is True
    assert result["checks"]["new_loss_improves_obs"] is True
    assert result["checks"]["obs_capacity_improves_obs"] is True
