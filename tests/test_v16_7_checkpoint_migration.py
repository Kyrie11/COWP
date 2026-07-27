from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def _tiny_model() -> torch.nn.Module:
    return torch.nn.ModuleDict({
        "graph": torch.nn.Linear(2, 2),
        "natural_decoder": torch.nn.Linear(2, 2),
        "set_transport": torch.nn.ModuleDict({
            "candidate_risk_raw_weight": torch.nn.Linear(2, 1, bias=False),
        }),
    })


def test_strict_migration_allows_only_explicit_new_transport_keys() -> None:
    train = importlib.import_module("cowp.scripts.03_train")
    model = _tiny_model()
    state = model.state_dict()
    missing_key = "set_transport.candidate_risk_raw_weight.weight"
    state = {k: v.clone() for k, v in state.items() if k != missing_key}
    report = train._strict_checkpoint_load(
        model, state, allowed_missing=frozenset({missing_key})
    )
    assert report["allowed_initialized_keys"] == [missing_key]


def test_strict_migration_rejects_missing_natural_key() -> None:
    train = importlib.import_module("cowp.scripts.03_train")
    model = _tiny_model()
    state = model.state_dict()
    state = {k: v.clone() for k, v in state.items() if k != "natural_decoder.weight"}
    with pytest.raises(RuntimeError, match="Strict checkpoint migration failed"):
        train._strict_checkpoint_load(model, state)


def test_learned_natural_loader_accepts_v16_7_downstream_additions(tmp_path: Path) -> None:
    diag = importlib.import_module("cowp.scripts.39_diagnose_learned_natural")

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph = torch.nn.Linear(2, 2)
            self.natural_decoder = torch.nn.Linear(2, 2)
            self.set_transport = torch.nn.Module()
            self.set_transport.candidate_risk_raw_weight = torch.nn.Parameter(torch.zeros(4))
            self.set_transport.candidate_risk_threshold_logit = torch.nn.Parameter(torch.zeros(()))
            self.set_transport.candidate_risk_log_scale = torch.nn.Parameter(torch.zeros(()))
            self.set_transport.global_risk_raw_weight = torch.nn.Parameter(torch.zeros(4))
            self.set_transport.global_risk_threshold_logit = torch.nn.Parameter(torch.zeros(()))
            self.set_transport.global_risk_log_scale = torch.nn.Parameter(torch.zeros(()))
            self.set_transport.pair_deficit_raw_weight = torch.nn.Parameter(torch.zeros(3))

    model = Toy()
    old_state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("set_transport.")}
    path = tmp_path / "natural_v16_6.pt"
    torch.save({"model": old_state, "epoch": 7, "stage": "natural"}, path)
    ckpt, report = diag._load_checkpoint(model, str(path), torch.device("cpu"))
    assert ckpt["epoch"] == 7
    assert set(report["allowed_initialized_keys"]) == diag._V16_7_NATURAL_DIAG_ALLOWED_MISSING


def test_learned_natural_loader_rejects_missing_natural_parameter(tmp_path: Path) -> None:
    diag = importlib.import_module("cowp.scripts.39_diagnose_learned_natural")

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.natural_decoder = torch.nn.Linear(2, 2)

    model = Toy()
    state = {k: v.clone() for k, v in model.state_dict().items() if k != "natural_decoder.weight"}
    path = tmp_path / "broken.pt"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Checkpoint/config mismatch"):
        diag._load_checkpoint(model, str(path), torch.device("cpu"))
