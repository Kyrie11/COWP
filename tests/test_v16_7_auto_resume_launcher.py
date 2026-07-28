from __future__ import annotations

import importlib
from pathlib import Path

import torch


resolver = importlib.import_module("cowp.scripts.46_resolve_training_checkpoint")


def _save(path: Path, *, stage: str, epoch: int) -> None:
    torch.save({"stage": stage, "epoch": epoch, "model": {}}, path)


def test_resolver_prefers_latest_epoch_and_last_on_tie(tmp_path: Path) -> None:
    _save(tmp_path / "cowp_planner_best.pt", stage="planner", epoch=2)
    _save(tmp_path / "cowp_planner_epoch004.pt", stage="planner", epoch=4)
    _save(tmp_path / "cowp_planner_last.pt", stage="planner", epoch=4)

    result = resolver.resolve_latest_checkpoint(tmp_path, "planner", target_epochs=10)
    assert Path(result.checkpoint).name == "cowp_planner_last.pt"
    assert result.epoch == 4
    assert result.next_epoch == 5
    assert result.complete is False


def test_resolver_marks_zero_based_target_as_complete(tmp_path: Path) -> None:
    _save(tmp_path / "cowp_witness_last.pt", stage="witness", epoch=13)
    result = resolver.resolve_latest_checkpoint(tmp_path, "witness", target_epochs=14)
    assert result.complete is True
    assert result.next_epoch == 14


def test_resolver_ignores_wrong_stage_and_corrupt_latest(tmp_path: Path) -> None:
    (tmp_path / "cowp_natural_last.pt").write_bytes(b"not a checkpoint")
    _save(tmp_path / "cowp_natural_epoch008.pt", stage="planner", epoch=8)
    _save(tmp_path / "cowp_natural_epoch006.pt", stage="natural", epoch=6)

    result = resolver.resolve_latest_checkpoint(tmp_path, "natural", target_epochs=20)
    assert Path(result.checkpoint).name == "cowp_natural_epoch006.pt"
    assert result.epoch == 6


def test_launcher_contains_same_stage_resume_for_all_training_stages() -> None:
    text = Path("run_cowp_v16_7_dual_gpu.sh").read_text(encoding="utf-8")
    assert "46_resolve_training_checkpoint.py" in text
    assert text.count("--resume-training") >= 3
    assert "stage_resume_state natural" in text
    assert "stage_resume_state witness" in text
    assert "stage_resume_state planner" in text
