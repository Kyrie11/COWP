from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss


ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "calibrate_policy_risk_v48", ROOT / "tools" / "calibrate_policy_risk_v48.py"
)
assert _spec is not None and _spec.loader is not None
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)


def _loss(rank: torch.Tensor, opp: torch.Tensor | None = None, harm: torch.Tensor | None = None, **extra) -> torch.Tensor:
    # Two complete groups in each regime are used so cross-group evidence pairs
    # can be formed.  Candidate 1 is beneficial in group 0, candidate 4 is
    # harmful in group 1; the other recovery is dead-zone.
    kwargs = dict(
        pred_logit=torch.zeros(6), pred_logvar=torch.zeros(6), pred_rank_logit=rank,
        pred_opportunity_logit=opp, pred_harm_logit=harm,
        teacher_r_dep=torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        teacher_r_orc=torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        teacher_q=torch.ones((6, 1, 1)),
        # Exact PCD targets: [0.5, 0.88, 0.5, 0.5, 0.0, 0.5].
        teacher_m_star=torch.tensor([
            [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[-1.0]], [[1.0]],
        ]),
        root_probs=torch.ones((6, 1)), root_valid=torch.ones((6, 1), dtype=torch.bool),
        option_valid=torch.ones((6, 1), dtype=torch.bool),
        scene_hash=torch.tensor([101, 101, 101, 202, 202, 202]),
        time_index=torch.tensor([1, 1, 1, 1, 1, 1]),
        macro_type_id=torch.tensor([0, 5, 3, 0, 5, 3]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1, 1, 1, 1]),
        macro_ids=(3, 5), bucket_ids=(1,), output_mode="score",
        exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
        advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0, setwise_admission_weight=0.0,
        policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0,
        preference_listwise_weight=0.0, preference_gap_weight=0.0,
        preference_set_weight=0.0, preference_all_group_set_weight=0.0,
        delta_nll_weight=0.0,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_gap_weighted_recovery_tournament_prefers_correct_order() -> None:
    good = _loss(
        torch.tensor([0.0, 2.0, -1.0, 0.0, -1.0, 2.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_pairwise_weight=1.0,
        preference_conditional_pairwise_min_gap=0.01,
    )
    bad = _loss(
        torch.tensor([0.0, -1.0, 2.0, 0.0, 2.0, -1.0]),
        preference_conditional_set_weight=1.0,
        preference_conditional_pairwise_weight=1.0,
        preference_conditional_pairwise_min_gap=0.01,
    )
    assert good.item() < bad.item()


def test_cross_group_ordinal_pairwise_loss_rewards_tail_separation() -> None:
    rank = torch.tensor([0.0, 2.0, -1.0, 0.0, 2.0, -1.0])
    # Group 0 selected candidate should be beneficial; group 1 selected candidate harmful.
    good_opp = torch.tensor([0.0, 3.0, 0.0, 0.0, -3.0, 0.0])
    good_harm = torch.tensor([0.0, -3.0, 0.0, 0.0, 3.0, 0.0])
    bad_opp = -good_opp
    bad_harm = -good_harm
    good = _loss(
        rank, good_opp, good_harm,
        ordinal_evidence_ordered_nll_top1_weight=1.0,
        ordinal_evidence_pairwise_benefit_weight=1.0,
        ordinal_evidence_pairwise_harm_weight=1.0,
    )
    bad = _loss(
        rank, bad_opp, bad_harm,
        ordinal_evidence_ordered_nll_top1_weight=1.0,
        ordinal_evidence_pairwise_benefit_weight=1.0,
        ordinal_evidence_pairwise_harm_weight=1.0,
    )
    assert good.item() < bad.item()


def test_opportunity_normalized_macro_concentration_reports_excess() -> None:
    groups = []
    top1 = []
    # Oracle-positive policy is 90% macro 5, 10% macro 3.  Selecting only
    # macro 5 has raw concentration 1.0 but excess concentration only 0.1.
    for i in range(10):
        macro = 5 if i < 9 else 3
        pair = {
            "candidate": 1, "macro": macro, "pred_adv": 1.0,
            "rank_margin": 1.0, "teacher_adv": 0.2,
        }
        groups.append({
            "scene": f"s{i}", "time": 0, "fold": 0,
            "oracle_best_teacher_adv": 0.2, "pairs": [pair],
        })
        if i < 9:
            top1.append(pair)
    # Add one selected macro-5 row so selected share is exactly 1.0.
    top1.append({
        "candidate": 2, "macro": 5, "pred_adv": 1.0,
        "rank_margin": 1.0, "teacher_adv": 0.2,
    })
    metrics = _cal._metrics(groups, top1, 0.0, 0.0, 0.015, 0.015)
    assert metrics["max_selected_macro_share"] == 1.0
    assert abs(metrics["oracle_positive_max_macro_share"] - 0.9) < 1e-8
    assert abs(metrics["selected_macro_excess_share"] - 0.1) < 1e-8


def test_stage_r_uses_conditional_checkpoint_semantics() -> None:
    text = (ROOT / "scripts" / "train_ocrap_v48_12_trident.sh").read_text()
    assert "PREFERENCE_CONDITIONAL_MODE=true" in text
