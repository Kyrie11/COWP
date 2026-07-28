from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def test_provenance_compatible_resume_preserves_initial_manifest(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.py"
    tracked.write_text("x=1\n", encoding="utf-8")
    out = tmp_path / "run_provenance.json"
    base = [
        sys.executable,
        "-m",
        "cowp.scripts.42_write_run_provenance",
        "--output",
        str(out),
        "--strict-existing",
        "--file",
        f"tracked.py={tracked}",
    ]
    subprocess.run(base, check=True, capture_output=True, text=True)
    old = json.loads(out.read_text(encoding="utf-8"))
    tracked.write_text("x=2\n", encoding="utf-8")
    subprocess.run(
        base + ["--allow-compatible-resume", "--resume-reason", "numeric hotfix"],
        check=True,
        capture_output=True,
        text=True,
    )
    new = json.loads(out.read_text(encoding="utf-8"))
    initial = tmp_path / "run_provenance.initial.json"
    amendments = tmp_path / "run_provenance_amendments.jsonl"
    assert initial.is_file() and amendments.is_file()
    assert json.loads(initial.read_text(encoding="utf-8"))["signature"] == old["signature"]
    assert new["resume_parent_signature"] == old["signature"]
    assert new["compatible_resume"] is True


def test_checkpoint_payload_carries_scheduler_and_resume_counter() -> None:
    import importlib

    train_mod = importlib.import_module("cowp.scripts.03_train")
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
    scheduler.step(2.0)
    payload = train_mod._make_checkpoint_payload(
        model,
        {"train": {}},
        opt,
        epoch=3,
        stage="planner",
        best_val=1.0,
        save_optimizer=True,
        scheduler=scheduler,
        no_improve_checks=2,
    )
    assert payload["epoch"] == 3
    assert payload["stage"] == "planner"
    assert payload["no_improve_checks"] == 2
    assert "optimizer" in payload and "scheduler" in payload


def test_atomic_checkpoint_write_leaves_no_partial_file(tmp_path: Path) -> None:
    import importlib

    train_mod = importlib.import_module("cowp.scripts.03_train")
    path = tmp_path / "cowp_planner_last.pt"
    train_mod._atomic_torch_save({"epoch": 4, "stage": "planner"}, path)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 4
    assert not list(tmp_path.glob("*.tmp.*"))


def test_legacy_plateau_scheduler_is_reconstructed_from_history() -> None:
    import importlib

    train_mod = importlib.import_module("cowp.scripts.03_train")
    scores = [2.0, 2.1, 2.2, 2.3, 2.4]
    history = [{"epoch": i, "checkpoint/score": score} for i, score in enumerate(scores)]

    ref_param = torch.nn.Parameter(torch.zeros(()))
    ref_opt = torch.optim.SGD([ref_param], lr=1e-3)
    ref_sched = train_mod._make_lr_scheduler(
        ref_opt,
        mode="plateau",
        epochs=10,
        early_stop_patience=4,
        min_lr=1e-6,
        min_delta=1e-4,
    )
    for score in scores:
        ref_sched.step(score)

    resumed_param = torch.nn.Parameter(torch.zeros(()))
    resumed_opt = torch.optim.AdamW([resumed_param], lr=7e-5)
    resumed_sched = train_mod._make_lr_scheduler(
        resumed_opt,
        mode="plateau",
        epochs=10,
        early_stop_patience=4,
        min_lr=1e-6,
        min_delta=1e-4,
    )
    replayed = train_mod._reconstruct_scheduler_from_history(
        resumed_sched,
        resumed_opt,
        history,
        mode="plateau",
        base_lr=1e-3,
        epochs=10,
        early_stop_patience=4,
        min_lr=1e-6,
        min_delta=1e-4,
    )
    assert replayed == len(scores)
    assert resumed_opt.param_groups[0]["lr"] == ref_opt.param_groups[0]["lr"]
    assert resumed_sched.state_dict() == ref_sched.state_dict()
