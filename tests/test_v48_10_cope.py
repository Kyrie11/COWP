from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


def _teacher_loss(rank: torch.Tensor, **extra) -> torch.Tensor:
    kwargs = dict(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3), pred_rank_logit=rank,
        teacher_r_dep=torch.tensor([-2.0, 2.0, -2.0]),
        teacher_r_orc=torch.tensor([-2.0, 2.0, -2.0]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.tensor([[[-1.0]], [[1.0]], [[-1.0]]]),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([101, 101, 101]), time_index=torch.tensor([1, 1, 1]),
        macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0, preference_listwise_weight=0.0,
        preference_gap_weight=0.0, preference_set_weight=0.0,
        preference_all_group_set_weight=0.0, delta_nll_weight=0.0,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_conditional_preference_ranks_recoveries_without_nominal_competition() -> None:
    good = _teacher_loss(
        torch.tensor([10.0, 2.0, -1.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_noop_weight=0.3,
    )
    bad = _teacher_loss(
        torch.tensor([-10.0, -1.0, 2.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_noop_weight=0.3,
    )
    # Nominal can be arbitrarily high or low; only the recovery ordering matters.
    assert good.item() < bad.item()


def test_ordinal_evidence_is_monotone_and_nominal_is_zero() -> None:
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True, direct_recovery_harm_head=True,
        direct_recovery_preference_context=True,
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_delta_head=True, direct_recovery_delta_hidden=16,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_initial_logvar=-2.0,
    ).eval()
    x = torch.randn(3, 12)
    group = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        out = model(x, group_index=group, is_nominal=nominal, direct_only=True)
    benefit = out["direct_recovery_evidence_benefit_logit"]
    nonharm = out["direct_recovery_evidence_nonharm_logit"]
    assert torch.all(benefit <= nonharm + 1.0e-7)
    assert float(benefit[0]) == 0.0
    assert float(nonharm[0]) == 0.0
    assert torch.allclose(out["direct_recovery_harm_logit"], -nonharm)


def test_policy_top1_ordinal_evidence_does_not_train_unused_candidate() -> None:
    rank = torch.tensor([0.0, 2.0, -1.0])
    opp = torch.zeros(3, requires_grad=True)
    harm = torch.zeros(3, requires_grad=True)
    loss = _teacher_loss(
        rank,
        pred_opportunity_logit=opp,
        pred_harm_logit=harm,
        ordinal_evidence_policy_top1_weight=2.0,
        ordinal_evidence_all_candidate_weight=0.0,
    )
    loss.backward()
    assert opp.grad is not None and harm.grad is not None
    assert abs(float(opp.grad[1])) > 0.0
    assert abs(float(harm.grad[1])) > 0.0
    assert float(opp.grad[2]) == 0.0
    assert float(harm.grad[2]) == 0.0


def test_conditional_rank_margin_excludes_nominal() -> None:
    path = Path(__file__).parents[1] / "tools" / "calibrate_policy_risk_v48.py"
    spec = importlib.util.spec_from_file_location("cal_v48_10", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    groups = [{
        "scene": "s", "time": 1, "fold": 0,
        "pairs": [
            {"candidate": 1, "macro": 5, "opportunity": 1.0, "harm": 0.0,
             "rank_adv": -0.10, "pred_adv": 0.5, "teacher_adv": 0.2},
            {"candidate": 2, "macro": 5, "opportunity": 1.0, "harm": 0.0,
             "rank_adv": -0.20, "pred_adv": 0.4, "teacher_adv": 0.1},
        ],
        "oracle_best_teacher_adv": 0.2,
    }]
    ordinary = mod._top1(groups, 0.5, 0.5, {5}, conditional_rank_margin=False)[0]
    conditional = mod._top1(groups, 0.5, 0.5, {5}, conditional_rank_margin=True)[0]
    assert ordinary["rank_margin"] < 0.0
    assert conditional["rank_margin"] > 0.0
