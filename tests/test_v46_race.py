from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.cli.train import _keep_fully_frozen_modules_in_eval
from ocrap.models.losses import direct_uncertainty_recovery_value_loss


def _direct_loss(pred: torch.Tensor, r_dep: torch.Tensor, *, negative_gain: float = 0.01) -> torch.Tensor:
    b = pred.numel()
    return direct_uncertainty_recovery_value_loss(
        pred_logit=pred,
        pred_logvar=torch.zeros_like(pred),
        teacher_r_dep=r_dep,
        teacher_r_orc=r_dep.clone(),
        teacher_q=torch.ones((b, 1, 1)),
        root_probs=torch.ones((b, 1)),
        root_valid=torch.ones((b, 1), dtype=torch.bool),
        option_valid=torch.ones((b, 1), dtype=torch.bool),
        scene_hash=torch.tensor([7, 7]),
        time_index=torch.tensor([3, 3]),
        macro_type_id=torch.tensor([0, 5]),
        is_nominal=torch.tensor([1.0, 0.0]),
        bucket_id=torch.tensor([1, 1]),
        macro_ids=(5,),
        bucket_ids=(1,),
        positive_gain=0.015,
        negative_gain=negative_gain,
        rank_margin=0.02,
        point_weight=0.0,
        listwise_weight=0.0,
        centered_weight=0.0,
        advantage_weight=0.0,
        pairwise_weight=1.0,
        ambiguous_group_weight=1.0,
        output_mode="score",
    )


def test_tied_teacher_advantage_is_regressed_to_tie_not_forced_negative() -> None:
    tied = torch.tensor([0.0, 0.0], requires_grad=True)
    forced_negative = torch.tensor([0.0, -0.2], requires_grad=True)
    teacher = torch.tensor([0.0, 0.0])
    assert _direct_loss(tied, teacher).item() < _direct_loss(forced_negative, teacher).item()


def test_meaningful_negative_teacher_advantage_rejects_false_positive() -> None:
    teacher = torch.tensor([5.0, -5.0])
    wrong = torch.tensor([0.0, 0.2], requires_grad=True)
    correct = torch.tensor([0.0, -0.2], requires_grad=True)
    assert _direct_loss(correct, teacher).item() < _direct_loss(wrong, teacher).item()


def test_fully_frozen_dropout_subtree_stays_in_eval() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
            self.head = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.Dropout(0.5))

    model = Toy()
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.train()
    _keep_fully_frozen_modules_in_eval(model)
    assert model.encoder.training is False
    assert model.encoder[1].training is False
    assert model.head.training is True
    assert model.head[1].training is True


def _load_calibration_module():
    path = Path(__file__).parents[1] / "tools" / "calibrate_direct_value_risk_v46.py"
    spec = importlib.util.spec_from_file_location("calibrate_v46", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_reports_conditional_harm_risk_and_ignores_ties() -> None:
    mod = _load_calibration_module()
    groups = [
        {"oracle_best_teacher_adv": 0.1},
        {"oracle_best_teacher_adv": 0.1},
        {"oracle_best_teacher_adv": 0.0},
        {"oracle_best_teacher_adv": 0.0},
    ]
    rows = [
        {"pred_adv": 0.2, "teacher_adv": -0.02, "opportunity": 0.9, "oracle_best_teacher_adv": 0.1},
        {"pred_adv": 0.2, "teacher_adv": 0.0, "opportunity": 0.9, "oracle_best_teacher_adv": 0.1},
    ]
    metrics = mod._metrics(groups, rows, 0.0, positive_gain=0.015, negative_gain=0.01)
    assert metrics["num_selected"] == 2
    assert metrics["num_harmful_selected"] == 1
    assert metrics["harmful_selected_rate"] == 0.5
    assert metrics["harmful_selected_ucb90"] > metrics["harmful_group_exposure_ucb90"]


def test_shared_raw_router_is_candidate_invariant() -> None:
    from ocrap.models.encoders import FlatFeatureLayout
    from ocrap.models.ocrap import OCRAPModel

    layout = FlatFeatureLayout(feature_max_agents=2)
    model = OCRAPModel(
        input_dim=layout.total_dim,
        num_roots=2,
        num_options=2,
        d_model=32,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout={"feature_max_agents": 2},
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat",
        direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True,
        direct_recovery_value_experts=True,
        direct_recovery_value_num_experts=2,
        direct_recovery_value_expert_routing="soft_observation",
        direct_recovery_value_router_pooling="shared_raw",
    ).eval()
    candidate_dim = model.direct_candidate_feature_dim
    x = torch.randn(1, layout.total_dim)
    changed = x.clone()
    changed[:, :candidate_dim] = torch.randn_like(changed[:, :candidate_dim]) * 10.0
    with torch.no_grad():
        a = model(x)["direct_expert_logits"]
        b = model(changed)["direct_expert_logits"]
    assert torch.equal(a, b)
