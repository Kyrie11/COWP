from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def test_training_rejects_nonfinite_predictions_before_loss() -> None:
    train_mod = importlib.import_module("cowp.scripts.03_train")
    pred = {
        "natural": {
            "traj": torch.tensor([0.0, float("nan")]),
            "residual": torch.tensor([0.0, float("inf")]),
        },
        "finite": torch.ones(2),
    }
    bad = train_mod._nonfinite_tensor_paths(pred)
    assert "pred.natural.traj" in bad
    assert "pred.natural.residual" in bad
    assert "pred.finite" not in bad


def test_amp_auto_prefers_bfloat16_when_supported(monkeypatch) -> None:
    train_mod = importlib.import_module("cowp.scripts.03_train")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert train_mod._resolve_amp_dtype(torch.device("cuda"), "auto") == torch.bfloat16


def test_v16_3_runner_keeps_natural_fp32_and_records_precision() -> None:
    path = Path("run_cowp_v16_3_dual_gpu.sh")
    if not path.exists():
        pytest.skip("legacy v16.3 launcher is not shipped in the active v16.8 package")
    source = path.read_text(encoding="utf-8")
    assert 'NATURAL_AMP="${NATURAL_AMP:-0}"' in source
    assert 'AMP_DTYPE="${AMP_DTYPE:-auto}"' in source
    assert "precision_manifest.json" in source
    assert "cowp/scripts/03_train.py=cowp/scripts/03_train.py" in source


def test_cnob_decoder_has_explicit_fp32_precision_island() -> None:
    source = Path("cowp/models/cowp_model.py").read_text(encoding="utf-8")
    assert "torch.autocast(device_type=device_type, enabled=False)" in source
    assert 'enc_scene["z_agent"].float()' in source


def test_cnob_zero_velocity_backward_is_finite() -> None:
    from cowp.models.natural_decoder import NaturalDecoder

    decoder = NaturalDecoder(
        d_model=16, modes=24, future_steps=8, decoder_type="typed_causal_dynamics"
    )
    z = torch.randn(1, 2, 16)
    idx = torch.tensor([[0]])
    # Exact zero velocity reproduces the first-step CNOB edge case.
    anchor = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 1.8]]])
    out = decoder(z, idx, anchor7=anchor, dt=0.1)
    objective = out["traj"][..., :5].square().mean() + out["controls"].square().mean()
    objective.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in decoder.parameters()
    )


def test_stable_grad_clip_handles_finite_fp32_norm_overflow() -> None:
    train_mod = importlib.import_module("cowp.scripts.03_train")
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.grad = torch.full_like(layer.weight, 3.0e38)
    norm, bad = train_mod._clip_grad_norm_stable(layer, 1.0)
    assert not bad
    assert torch.isfinite(norm)
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.linalg.vector_norm(layer.weight.grad).item() <= 1.0001


def test_recovery_wrapper_preserves_output_root_and_rotates_only_failed_provenance() -> None:
    path = Path("NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh")
    if not path.exists():
        pytest.skip("legacy v16.3 recovery wrapper is not shipped in the active v16.8 package")
    source = path.read_text(encoding="utf-8")
    assert 'OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_3_natural_recovery_v9labels_seed2026}"' in source
    assert 'mv "$PROVENANCE" "${PROVENANCE}.failed_before_numeric_fix.${stamp}.bak"' in source
    assert '! -s "$NATURAL_HISTORY"' in source
