from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss


def _base_loss(*, rank: torch.Tensor, delta: torch.Tensor | None = None, **extra) -> torch.Tensor:
    kwargs = dict(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(3), pred_rank_logit=rank,
        pred_delta_mean=delta,
        teacher_r_dep=torch.tensor([-2.0, 2.0, -2.0]),
        teacher_r_orc=torch.tensor([-2.0, 2.0, -2.0]),
        teacher_q=torch.ones((3, 1, 1)),
        teacher_m_star=torch.tensor([[[-1.0]], [[1.0]], [[-1.0]]]),
        root_probs=torch.ones((3, 1)), root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([99, 99, 99]), time_index=torch.tensor([1, 1, 1]),
        macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
        output_mode="score", exact_teacher_pcd=True, positive_gain=0.01, negative_gain=0.01,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0, advantage_weight=0.0,
        pairwise_weight=0.0, top_rank_weight=0.0, opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=0.0, policy_distill_weight=0.0, policy_regret_weight=0.0,
        preference_weight=0.0, preference_regret_weight=0.0, preference_listwise_weight=0.0,
        preference_gap_weight=0.0, preference_set_weight=0.0, delta_nll_weight=0.0,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_policy_aligned_certificate_updates_only_frozen_policy_top1() -> None:
    rank = torch.tensor([0.0, 1.0, -1.0])
    delta = torch.zeros(3, requires_grad=True)
    loss = _base_loss(
        rank=rank, delta=delta,
        certificate_policy_top1_weight=2.0,
        certificate_policy_top1_sign_weight=1.0,
    )
    loss.backward()
    assert delta.grad is not None
    assert abs(float(delta.grad[1])) > 0.0
    assert float(delta.grad[2]) == 0.0


def test_partial_label_set_mass_does_not_force_uniform_acceptable_logits() -> None:
    # Candidate 1 is the only material positive in this synthetic group.  The
    # set-mass objective should strongly prefer concentrating on it.
    good = _base_loss(
        rank=torch.tensor([0.0, 2.0, -1.0]),
        preference_all_group_set_weight=1.0,
        preference_set_replace_singlewinner=True,
        preference_set_mass_loss=True,
        preference_noop_nominal_only=True,
    )
    bad = _base_loss(
        rank=torch.tensor([0.0, -1.0, 2.0]),
        preference_all_group_set_weight=1.0,
        preference_set_replace_singlewinner=True,
        preference_set_mass_loss=True,
        preference_noop_nominal_only=True,
    )
    assert good.item() < bad.item()


def test_policy_top1_conformal_scope_uses_one_candidate_per_group() -> None:
    path = Path(__file__).parents[1] / "tools" / "calibrate_policy_risk_v48.py"
    spec = importlib.util.spec_from_file_location("cal_v48_pacer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    groups = [
        {"scene": "a", "time": 1, "fold": 0, "pairs": [
            {"candidate": 1, "rank_adv": 0.1, "pred_adv": 0.1, "teacher_adv": 0.0},
            {"candidate": 2, "rank_adv": 0.4, "pred_adv": 0.2, "teacher_adv": 0.3},
        ]},
        {"scene": "b", "time": 2, "fold": 0, "pairs": [
            {"candidate": 1, "rank_adv": -0.1, "pred_adv": -0.1, "teacher_adv": -0.2},
        ]},
    ]
    rows = mod._policy_top1_pairs(groups)
    assert len(rows) == 2
    assert rows[0]["candidate"] == 2
