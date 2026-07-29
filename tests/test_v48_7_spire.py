from __future__ import annotations

import torch

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.losses import direct_uncertainty_recovery_value_loss


def _set_loss(rank: list[float]) -> torch.Tensor:
    return direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor(rank, dtype=torch.float32),
        teacher_r_dep=torch.tensor([-2.0, 2.0, 0.5]),
        teacher_r_orc=torch.tensor([-2.0, 2.0, 0.5]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.tensor([[[-1.0]], [[1.0]], [[1.0]]]),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([71, 71, 71]), time_index=torch.tensor([1, 1, 1]),
        macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0, preference_listwise_weight=0.0,
        preference_gap_weight=0.0, preference_set_weight=1.0,
        preference_tie_epsilon_near=1.0, preference_set_margin=0.02,
    )


def test_set_valued_preference_does_not_penalize_arbitrary_tie_order() -> None:
    first = _set_loss([0.0, 1.0, 0.8])
    second = _set_loss([0.0, 0.8, 1.0])
    assert abs(float(first.item() - second.item())) < 1e-5


def test_certificate_metric_penalizes_always_abstain_on_positive_groups() -> None:
    out = _finalize_direct_policy_stats({
        "group_count_near": 10.0,
        "positive_count_near": 5.0,
        "positive_regret_sum_near": 0.0,
        "positive_top1_hit_near": 5.0,
        "positive_admission_hit_near": 0.0,
        "positive_rank_margin_sum_near": 1.0,
        "admission_count_near": 0.0,
        "admitted_harmful_near": 0.0,
        "false_intervention_near": 0.0,
    }, {"direct_policy_metric_missed_opportunity_weight": 0.25})
    assert out["direct_certificate_risk_mean_near"] >= 0.25


def test_preference_only_negative_batch_is_zero_gradient_noop() -> None:
    """A legal batch without a positive recovery group must still backpropagate.

    Stage P freezes the value/certificate branches and trains only the ranking
    residuals.  Negative-only batches have no active preference term, so their
    loss is exactly zero; backward must produce zero gradients instead of raising
    ``element 0 of tensors does not require grad``.
    """
    rank = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float32, requires_grad=True)
    loss = direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3),
        pred_rank_logit=rank,
        teacher_r_dep=torch.tensor([2.0, -2.0, -1.0]),
        teacher_r_orc=torch.tensor([2.0, -2.0, -1.0]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.tensor([[[1.0]], [[-1.0]], [[-1.0]]]),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([91, 91, 91]), time_index=torch.tensor([2, 2, 2]),
        macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True,
        positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
        advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0, setwise_admission_weight=0.0,
        policy_distill_weight=0.0, policy_regret_weight=0.0,
        policy_admission_distill_weight=0.0,
        preference_weight=1.0, preference_regret_weight=0.75,
        preference_listwise_weight=0.0, preference_gap_weight=0.15,
        preference_set_weight=1.25, delta_nll_weight=0.0,
    )
    assert loss.requires_grad
    loss.backward()
    assert rank.grad is not None
    assert torch.count_nonzero(rank.grad) == 0


def test_certificate_only_filtered_batch_is_zero_gradient_noop() -> None:
    """Stage C must tolerate a batch with no allowed recovery macro."""
    delta_mean = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    delta_logvar = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    loss = direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3),
        pred_delta_mean=delta_mean, pred_delta_logvar=delta_logvar,
        teacher_r_dep=torch.tensor([0.0, 1.0, -1.0]),
        teacher_r_orc=torch.tensor([0.0, 1.0, -1.0]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.ones((3, 1, 1)),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([92, 92, 92]), time_index=torch.tensor([3, 3, 3]),
        macro_type_id=torch.tensor([0, 3, 3]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([2, 2, 2]), macro_ids=(5,), bucket_ids=(2,),
        output_mode="score", exact_teacher_pcd=True,
        point_weight=0.0, centered_weight=1.0, listwise_weight=0.0,
        advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0, setwise_admission_weight=0.0,
        policy_distill_weight=0.0, policy_regret_weight=0.0,
        policy_admission_distill_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0,
        preference_listwise_weight=0.0, preference_gap_weight=0.0,
        preference_set_weight=0.0, delta_nll_weight=1.0,
    )
    assert loss.requires_grad
    loss.backward()
    assert delta_mean.grad is not None
    assert delta_logvar.grad is not None
    assert torch.count_nonzero(delta_mean.grad) == 0
    assert torch.count_nonzero(delta_logvar.grad) == 0
