from __future__ import annotations

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


def _model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=12,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="mlp",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_output="score",
        direct_recovery_preference_head=True,
        direct_recovery_preference_dropout=0.0,
        direct_recovery_preference_context=True,
        direct_recovery_preference_context_hidden=16,
        direct_recovery_delta_head=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0,
    ).eval()


def test_rpgc_new_heads_are_warm_start_safe_and_nominal_anchored() -> None:
    torch.manual_seed(61)
    model = _model()
    x = torch.randn(3, 12)
    group = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        out = model(x, group_index=group, is_nominal=nominal, direct_only=True)
    # Both new residual projections are zero-initialized.  The inherited
    # preference score is therefore preserved exactly at the first step.
    assert torch.allclose(out["direct_recovery_rank_logit"], out["direct_recovery_value_logit"], atol=1e-8)
    assert torch.count_nonzero(out["direct_recovery_rank_context_residual"]) == 0
    assert torch.count_nonzero(out["direct_recovery_delta_mean"]) == 0
    assert out["direct_recovery_delta_mean"][0].item() == 0.0


def test_preference_context_is_permutation_equivariant_after_learning() -> None:
    torch.manual_seed(62)
    model = _model()
    assert model.direct_preference_context_adapter is not None
    with torch.no_grad():
        projection = model.direct_preference_context_adapter[-1]
        projection.weight.normal_(mean=0.0, std=0.1)
        projection.bias.zero_()
    x = torch.randn(4, 12)
    group = torch.zeros((4, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0])
    perm = torch.tensor([0, 3, 1, 2])
    with torch.no_grad():
        base = model(x, group_index=group, is_nominal=nominal, direct_only=True)["direct_recovery_rank_logit"]
        shuffled = model(
            x[perm], group_index=group, is_nominal=nominal[perm], direct_only=True
        )["direct_recovery_rank_logit"]
    assert torch.allclose(base, shuffled[torch.argsort(perm)], atol=1e-6)


def _delta_loss(delta: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3),
        pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 1.0, 0.1]),
        pred_delta_mean=delta,
        pred_delta_logvar=logvar,
        teacher_r_dep=torch.tensor([-2.0, 2.0, 0.5]),
        teacher_r_orc=torch.tensor([-2.0, 2.0, 0.5]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.tensor([[[-1.0]], [[1.0]], [[1.0]]]),
        root_probs=torch.ones((3, 1)),
        root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([61, 61, 61]),
        time_index=torch.tensor([2, 2, 2]),
        macro_type_id=torch.tensor([0, 5, 5]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([2, 2, 2]),
        macro_ids=(5,),
        bucket_ids=(2,),
        output_mode="score",
        exact_teacher_pcd=True,
        positive_gain=0.01,
        negative_gain=0.01,
        point_weight=0.0,
        centered_weight=1.0,
        delta_nll_weight=1.0,
        listwise_weight=0.0,
        advantage_weight=0.0,
        pairwise_weight=0.0,
        top_rank_weight=0.0,
        opportunity_weight=0.0,
        harm_weight=0.0,
        setwise_admission_weight=0.0,
        policy_distill_weight=0.0,
        policy_regret_weight=0.0,
        preference_weight=0.0,
        preference_regret_weight=0.0,
    )


def test_direct_relative_gain_head_is_supervised_by_exact_candidate_minus_nominal() -> None:
    # Candidate 1 is the strong positive recovery and candidate 2 is weaker.
    correct = _delta_loss(torch.tensor([0.0, 0.8, 0.2]), torch.full((3,), -3.0))
    reversed_delta = _delta_loss(torch.tensor([0.0, 0.2, 0.8]), torch.full((3,), -3.0))
    assert correct.item() + 0.05 < reversed_delta.item()
