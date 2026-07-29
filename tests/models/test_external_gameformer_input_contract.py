from __future__ import annotations

import torch

from ocrap.external_baselines.data import use_teacher_branch_context
from ocrap.external_baselines.train import _forward_model, train_external_baseline


class _CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, x, mask, **kwargs):
        self.kwargs = kwargs
        return {"logits": torch.zeros(x.shape[0], x.shape[1], device=x.device)}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros(2, 3, 4),
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "branch_margins": torch.ones(2, 3, 2, 2),
        "root_features": torch.ones(2, 3, 2, 5),
        "root_probs": torch.full((2, 3, 2), 0.5),
        "root_valid": torch.ones(2, 3, 2, dtype=torch.bool),
        "option_valid": torch.ones(2, 3, 2, dtype=torch.bool),
    }


def test_deployable_input_contract_masks_teacher_branch_tensors() -> None:
    cfg = {"external_baselines": {"model": {"use_teacher_branch_context": False}}}
    model = _CaptureModel()
    _forward_model(model, _batch(), cfg)
    assert use_teacher_branch_context(cfg) is False
    assert model.kwargs is not None
    assert model.kwargs["branch_margins"] is None
    assert model.kwargs["root_features"] is None
    assert model.kwargs["root_probs"] is None
    assert model.kwargs["root_valid"] is None


def test_teacher_branch_context_is_opt_in_only() -> None:
    cfg = {"external_baselines": {"model": {}}}
    model = _CaptureModel()
    _forward_model(model, _batch(), cfg)
    assert use_teacher_branch_context(cfg) is False
    assert model.kwargs is not None
    assert model.kwargs["branch_margins"] is None

    diagnostic_cfg = {"external_baselines": {"model": {"use_teacher_branch_context": True}}}
    batch = _batch()
    diagnostic_model = _CaptureModel()
    _forward_model(diagnostic_model, batch, diagnostic_cfg)
    assert use_teacher_branch_context(diagnostic_cfg) is True
    assert diagnostic_model.kwargs["branch_margins"] is batch["branch_margins"]


def test_nonlearning_registration_can_skip_redundant_dataset_scan(tmp_path) -> None:
    cfg = {
        "external_baselines": {
            "baseline": "marc_lite",
            "max_candidates": 24,
            "training": {"validate_dataset": False},
        }
    }
    summary = train_external_baseline(
        "/path/that/does/not/exist",
        str(tmp_path / "marc_lite"),
        cfg,
        baseline="marc_lite",
    )
    assert summary["dataset_validated"] is False
    assert summary["num_train_groups"] is None
    assert (tmp_path / "marc_lite" / "train_summary.json").exists()
