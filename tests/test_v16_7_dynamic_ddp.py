from __future__ import annotations

import importlib

import torch


def test_dynamic_supervision_stages_never_enable_static_graph() -> None:
    train = importlib.import_module("cowp.scripts.03_train")
    device = torch.device("cpu")
    for stage in ("response", "witness", "planner", "all"):
        policy = train._ddp_policy(stage, device=device, local_rank=0)
        assert policy["find_unused_parameters"] is True
        assert policy.get("static_graph", False) is False


def test_natural_stage_does_not_claim_static_graph_without_proof() -> None:
    train = importlib.import_module("cowp.scripts.03_train")
    policy = train._ddp_policy("natural", device=torch.device("cpu"), local_rank=0)
    assert policy["find_unused_parameters"] is False
    assert policy.get("static_graph", False) is False


def test_permanent_witness_freeze_is_applied_before_ddp() -> None:
    train = importlib.import_module("cowp.scripts.03_train")

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph = torch.nn.Linear(2, 2)
            self.candidate_encoder = torch.nn.Linear(2, 2)
            self.natural_decoder = torch.nn.Linear(2, 2)
            self.witness_decoder = torch.nn.Linear(2, 2)

    model = Toy()
    train._set_stage_freeze(
        model,
        "witness",
        False,
        freeze_natural_during_witness=True,
        freeze_graph_during_natural=False,
    )
    assert all(p.requires_grad for p in model.graph.parameters())
    assert all(p.requires_grad for p in model.candidate_encoder.parameters())
    assert all(not p.requires_grad for p in model.natural_decoder.parameters())
    assert all(p.requires_grad for p in model.witness_decoder.parameters())


def test_ddp_manifest_indices_match_trainable_parameter_order() -> None:
    train = importlib.import_module("cowp.scripts.03_train")
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].bias.requires_grad_(False)
    policy = train._ddp_policy("witness", device=torch.device("cpu"), local_rank=0)
    manifest = train._ddp_parameter_manifest(model, "witness", policy)
    names = [row["name"] for row in manifest["parameters"]]
    expected = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names == expected
    assert [row["index"] for row in manifest["parameters"]] == list(range(len(expected)))
