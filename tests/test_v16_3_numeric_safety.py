from __future__ import annotations

import importlib
from pathlib import Path

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
    source = Path("run_cowp_v16_3_dual_gpu.sh").read_text(encoding="utf-8")
    assert 'NATURAL_AMP="${NATURAL_AMP:-0}"' in source
    assert 'AMP_DTYPE="${AMP_DTYPE:-auto}"' in source
    assert "precision_manifest.json" in source
    assert "cowp/scripts/03_train.py=cowp/scripts/03_train.py" in source


def test_cnob_decoder_has_explicit_fp32_precision_island() -> None:
    source = Path("cowp/models/cowp_model.py").read_text(encoding="utf-8")
    assert "torch.autocast(device_type=device_type, enabled=False)" in source
    assert 'enc_scene["z_agent"].float()' in source
