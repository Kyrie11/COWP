from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from torch import nn


class _ToyNaturalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Linear(2, 3)
        self.natural_decoder = nn.Linear(3, 4)
        self.set_transport = nn.Module()
        for name, value in {
            "candidate_risk_raw_weight": 1.0,
            "candidate_risk_threshold_logit": 2.0,
            "candidate_risk_log_scale": 3.0,
            "global_risk_raw_weight": 4.0,
            "global_risk_threshold_logit": 5.0,
            "global_risk_log_scale": 6.0,
            "pair_deficit_raw_weight": 7.0,
        }.items():
            self.set_transport.register_parameter(name, nn.Parameter(torch.tensor(value)))


def _diagnose_module():
    return importlib.import_module("cowp.scripts.39_diagnose_learned_natural")


def test_learned_natural_accepts_v16_6_checkpoint_missing_only_downstream_keys(tmp_path: Path) -> None:
    source = _ToyNaturalModel()
    old_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("set_transport.")
    }
    checkpoint = tmp_path / "v16_6_natural.pt"
    torch.save({"model": old_state, "stage": "natural", "epoch": 12}, checkpoint)

    loaded = _ToyNaturalModel()
    _diagnose_module()._load_checkpoint(loaded, str(checkpoint), torch.device("cpu"))

    for key, value in old_state.items():
        assert torch.equal(loaded.state_dict()[key], value)
    # New v16.7 downstream-only parameters retain current-config initialization.
    assert loaded.set_transport.pair_deficit_raw_weight.item() == pytest.approx(7.0)


def test_learned_natural_still_rejects_missing_natural_dependency(tmp_path: Path) -> None:
    source = _ToyNaturalModel()
    old_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("set_transport.")
    }
    old_state.pop("natural_decoder.weight")
    checkpoint = tmp_path / "broken_natural.pt"
    torch.save({"model": old_state, "stage": "natural", "epoch": 12}, checkpoint)

    with pytest.raises(RuntimeError, match="natural_decoder.weight"):
        _diagnose_module()._load_checkpoint(
            _ToyNaturalModel(), str(checkpoint), torch.device("cpu")
        )


def test_launcher_uses_exact_stage_directories_and_same_stage_resume() -> None:
    launcher = Path("run_cowp_v16_7_dual_gpu.sh").read_text(encoding="utf-8")
    assert 'stage_training_plan "$OUT_ROOT/checkpoints/natural" natural' in launcher
    assert 'stage_training_plan "$OUT_ROOT/checkpoints/transport" witness' in launcher
    assert 'stage_training_plan "$OUT_ROOT/checkpoints/planner" planner' in launcher
    assert 'resume_args=(--resume "$transport_resume" --resume-training)' in launcher
    assert 'resume_args=(--resume "$planner_resume" --resume-training)' in launcher
    assert '--resume "$natural_resume" --resume-training' in launcher
    assert 'FORCE_RESTART_TRAIN="${FORCE_RESTART_TRAIN:-0}"' in launcher
    wrapper = Path("NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh").read_text(encoding="utf-8")
    assert ".v16_7_dynamic_ddp_hotfix_v3_applied" in wrapper
    assert 'export ALLOW_COMPATIBLE_CODE_RESUME=1' in wrapper
