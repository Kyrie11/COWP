from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


def _scope_loss(rank: torch.Tensor, *, positive: bool) -> torch.Tensor:
    if positive:
        m = torch.tensor([[[-1.0]], [[1.0]], [[1.0]]])
        dep = torch.tensor([-2.0, 2.0, 1.8])
    else:
        m = torch.tensor([[[1.0]], [[-1.0]], [[-1.0]]])
        dep = torch.tensor([2.0, -2.0, -1.0])
    return direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3), pred_rank_logit=rank,
        teacher_r_dep=dep, teacher_r_orc=dep, teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=m, root_probs=torch.ones((3, 1)),
        root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([801, 801, 801]), time_index=torch.tensor([1, 1, 1]),
        macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0, preference_listwise_weight=0.0,
        preference_gap_weight=0.0, preference_set_weight=0.0,
        preference_all_group_set_weight=1.0, preference_set_replace_singlewinner=True,
        preference_tie_epsilon_near=0.03, preference_nominal_margin=0.02,
        preference_harm_margin=0.03,
    )


def test_all_group_set_prefers_recovery_only_when_materially_positive() -> None:
    positive_good = _scope_loss(torch.tensor([0.0, 1.0, 0.9]), positive=True)
    positive_nominal = _scope_loss(torch.tensor([1.0, 0.0, -0.1]), positive=True)
    negative_good = _scope_loss(torch.tensor([1.0, 0.0, -0.1]), positive=False)
    negative_switch = _scope_loss(torch.tensor([0.0, 1.0, 0.9]), positive=False)
    assert positive_good.item() < positive_nominal.item()
    assert negative_good.item() < negative_switch.item()


def test_relative_only_context_uses_three_feature_blocks_and_is_equivariant() -> None:
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_output="score",
        direct_recovery_preference_head=True, direct_recovery_preference_context=True,
        direct_recovery_preference_context_hidden=16,
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_delta_head=True, direct_recovery_delta_hidden=16,
    ).eval()
    assert model.direct_preference_context_adapter is not None
    assert model.direct_preference_context_adapter[0].normalized_shape[0] == 3 * 8
    with torch.no_grad():
        model.direct_preference_context_adapter[-1].weight.normal_(0.0, 0.1)
    x=torch.randn(4,12); group=torch.zeros((4,1),dtype=torch.long); nominal=torch.tensor([1.,0.,0.,0.])
    perm=torch.tensor([0,3,1,2])
    with torch.no_grad():
        a=model(x,group_index=group,is_nominal=nominal,direct_only=True)["direct_recovery_rank_logit"]
        b=model(x[perm],group_index=group,is_nominal=nominal[perm],direct_only=True)["direct_recovery_rank_logit"]
    assert torch.allclose(a,b[torch.argsort(perm)],atol=1e-6)


def test_support_aware_fold_metric_ignores_unsupported_singleton_fold() -> None:
    stats={
        "group_count_near_fold0":20., "positive_count_near_fold0":10.,
        "positive_regret_sum_near_fold0":1., "positive_top1_hit_near_fold0":8.,
        "group_count_contact_fold0":20., "positive_count_contact_fold0":10.,
        "positive_regret_sum_contact_fold0":2., "positive_top1_hit_contact_fold0":7.,
        "group_count_near_fold1":1., "positive_count_near_fold1":1.,
        "positive_regret_sum_near_fold1":1., "positive_top1_hit_near_fold1":0.,
    }
    out=_finalize_direct_policy_stats(stats,{"direct_policy_metric_min_fold_positive":6,"direct_policy_metric_robust_top_k":2})
    assert out["direct_preference_risk_supported_fold_count"] == 2.0
    assert out["direct_preference_risk_fold_robust"] < out["direct_preference_risk_fold_worst"]


def test_finite_sample_conformal_quantile_is_conservative() -> None:
    path=Path(__file__).parents[1]/"tools"/"calibrate_policy_risk_v48.py"
    spec=importlib.util.spec_from_file_location("cal_v48",path); mod=importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(mod)
    assert mod._finite_sample_upper_quantile([0.1,0.2,0.3,0.4],0.10) == 0.4
