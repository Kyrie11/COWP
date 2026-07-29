from __future__ import annotations

import numpy as np
import torch

from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select


def _risk_select(predicted_harm: float):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.05]),
        r_dep=np.array([0.5, -1.0]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.zeros(2),
        pred_drs=np.ones(2),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([0.0, 0.8]),
        pred_direct_std=np.zeros(2),
        pred_direct_opportunity=np.array([0.5, 0.9]),
        pred_direct_harm=np.array([0.0, predicted_harm]),
        candidate_macro_names=["nominal", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="yield",
        direct_value_uncertainty_mode="risk_controlled",
        direct_value_min_advantage_lcb=0.5,
        direct_value_score_mode=True,
        direct_value_opportunity_threshold=0.7,
        direct_value_harm_threshold=0.25,
        direct_value_top1_only=True,
        direct_value_risk_controlled_admission=True,
        direct_value_challenge_nominal=True,
        direct_value_bonus=1.0,
        stress_rescue_challenge_nominal=True,
    )


def test_predicted_harm_is_a_real_selector_gate():
    assert _risk_select(0.10).selected_index == 1
    assert _risk_select(0.80).selected_index == 0


def test_robust_expert_aggregation_is_conservative_and_bucket_free():
    model = OCRAPModel(
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
        direct_recovery_value_pooling="scene",
        direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True,
        direct_recovery_harm_head=True,
        direct_recovery_value_experts=True,
        direct_recovery_value_num_experts=2,
        direct_recovery_value_expert_routing="robust_ensemble",
        direct_recovery_expert_disagreement_penalty=0.5,
    ).eval()
    assert model.direct_value_heads is not None
    with torch.no_grad():
        for head in model.direct_value_heads:
            for param in head.parameters():
                param.zero_()
        model.direct_value_heads[0][-1].bias.copy_(torch.tensor([2.0, 0.0, 2.0, -2.0]))
        model.direct_value_heads[1][-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))
        x = torch.zeros(2, 12)
        out = model(x, bucket_id=torch.tensor([1, 2]), direct_only=True)
    # mean +/- 0.5 std: gain/opportunity lower bound, harm upper bound.
    assert torch.allclose(out["direct_recovery_value_logit"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out["direct_recovery_opportunity_logit"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out["direct_recovery_harm_logit"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(out["direct_expert_weights"], torch.full((2, 2), 0.5))


def _src_loss(pred: torch.Tensor, *, risk_weight: float) -> torch.Tensor:
    from ocrap.models.losses import direct_uncertainty_recovery_value_loss

    # nominal, teacher-positive recovery, teacher-harmful recovery
    return direct_uncertainty_recovery_value_loss(
        pred_logit=pred,
        pred_logvar=torch.zeros_like(pred),
        teacher_r_dep=torch.tensor([0.0, 2.0, -2.0]),
        teacher_r_orc=torch.tensor([0.0, 2.0, -2.0]),
        teacher_q=torch.ones((3, 1, 1)),
        root_probs=torch.ones((3, 1)),
        root_valid=torch.ones((3, 1), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([17, 17, 17]),
        time_index=torch.tensor([5, 5, 5]),
        macro_type_id=torch.tensor([0, 5, 5]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.tensor([1, 1, 1]),
        macro_ids=(5,), bucket_ids=(1,), output_mode="score",
        positive_gain=0.10, negative_gain=0.10, temperature=0.10,
        point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
        advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0,
        setwise_admission_weight=1.0e-6,
        selective_risk_weight=risk_weight,
        selective_harm_budget=0.05,
        selective_coverage_weight=0.0,
    )


def test_selective_risk_penalizes_policy_mass_on_harmful_candidate() -> None:
    positive_high = torch.tensor([0.0, 1.0, -1.0])
    harmful_high = torch.tensor([0.0, -1.0, 1.0])
    base_gap = _src_loss(harmful_high, risk_weight=0.0) - _src_loss(positive_high, risk_weight=0.0)
    risk_gap = _src_loss(harmful_high, risk_weight=4.0) - _src_loss(positive_high, risk_weight=4.0)
    assert risk_gap.item() > base_gap.item() + 0.5


def test_sampler_uses_canonical_positive_advantage_config_names(tmp_path) -> None:
    import json
    from types import SimpleNamespace
    from ocrap.cli.train import _make_group_batch_sampler

    paths = [tmp_path / f"candidate_{i}.npz" for i in range(4)]
    for path in paths:
        path.touch()
    rows = [
        {"path": str(paths[0]), "scene": "positive", "time": 0, "bucket": 1, "nominal": True, "macro": 0, "teacher_pcd": 0.10},
        {"path": str(paths[1]), "scene": "positive", "time": 0, "bucket": 1, "nominal": False, "macro": 5, "teacher_pcd": 0.40},
        {"path": str(paths[2]), "scene": "negative", "time": 0, "bucket": 1, "nominal": True, "macro": 0, "teacher_pcd": 0.30},
        {"path": str(paths[3]), "scene": "negative", "time": 0, "bucket": 1, "nominal": False, "macro": 5, "teacher_pcd": 0.20},
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = {"training": {
        "group_batching": True,
        "group_index_path": str(index),
        "group_batching_replacement": False,
        "group_batch_positive_advantage_boost": 5.0,
        "group_batch_positive_advantage_gain_min": 0.05,
        "group_batch_positive_advantage_macro_ids": "5",
        "group_batch_positive_advantage_bucket_ids": "1",
        "group_batch_require_positive_advantage_groups": True,
        "artifact_sampler_weight": 0.0,
        "negative_deployable_sampler_weight": 0.0,
        "safe_positive_sampler_weight": 0.0,
        "regime_balance_power": 0.0,
    }}
    sampler = _make_group_batch_sampler(SimpleNamespace(paths=paths), cfg, batch_size=4)
    assert sampler is not None
    assert sorted(sampler.group_weights.tolist()) == [1.0, 5.0]


def test_encoder_anchor_is_zero_then_increases_after_drift() -> None:
    from ocrap.cli.train import _parameter_anchor_loss
    layer = torch.nn.Linear(3, 2)
    layer._encoder_anchor_tensors = {
        "weight": layer.weight.detach().clone(),
        "bias": layer.bias.detach().clone(),
    }
    assert _parameter_anchor_loss(layer).item() == 0.0
    with torch.no_grad():
        layer.weight.add_(0.5)
    assert _parameter_anchor_loss(layer).item() > 0.0


def test_nasc_set_context_is_permutation_equivariant_and_singleton_safe() -> None:
    torch.manual_seed(5)
    model = OCRAPModel(
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
        direct_recovery_value_pooling="scene",
        direct_recovery_value_output="score",
        direct_recovery_set_context=True,
        direct_recovery_set_context_hidden=16,
        direct_recovery_set_context_dropout=0.0,
    ).eval()
    x = torch.randn(3, 12)
    group = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        base = model(x, group_index=group, is_nominal=nominal, direct_only=True)["direct_recovery_value_logit"]
        perm = torch.tensor([0, 2, 1])
        shuffled = model(
            x[perm], group_index=group, is_nominal=nominal[perm], direct_only=True
        )["direct_recovery_value_logit"]
        inverse = torch.argsort(perm)
        singleton = model(
            x[:1], group_index=torch.zeros((1, 1), dtype=torch.long),
            is_nominal=torch.ones(1), direct_only=True,
        )["direct_recovery_value_logit"]
        pointwise = model(x[:1], direct_only=True)["direct_recovery_value_logit"]
    assert torch.allclose(base, shuffled[inverse], atol=1.0e-6)
    assert torch.allclose(singleton, pointwise, atol=1.0e-6)


def test_regret_consistent_distillation_prefers_teacher_best_candidate() -> None:
    def loss(pred: torch.Tensor) -> torch.Tensor:
        from ocrap.models.losses import direct_uncertainty_recovery_value_loss
        return direct_uncertainty_recovery_value_loss(
            pred_logit=pred,
            pred_logvar=torch.zeros_like(pred),
            teacher_r_dep=torch.tensor([0.0, 2.0, 1.0]),
            teacher_r_orc=torch.tensor([0.0, 2.0, 1.0]),
            teacher_q=torch.ones((3, 1, 1)),
            root_probs=torch.ones((3, 1)),
            root_valid=torch.ones((3, 1), dtype=torch.bool),
            option_valid=torch.ones((3, 1), dtype=torch.bool),
            scene_hash=torch.tensor([19, 19, 19]),
            time_index=torch.tensor([3, 3, 3]),
            macro_type_id=torch.tensor([0, 5, 5]),
            is_nominal=torch.tensor([1.0, 0.0, 0.0]),
            bucket_id=torch.tensor([1, 1, 1]),
            macro_ids=(5,), bucket_ids=(1,), output_mode="score",
            positive_gain=0.10, negative_gain=0.10, temperature=0.10,
            point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
            advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
            opportunity_weight=0.0, harm_weight=0.0,
            setwise_admission_weight=1.0e-6,
            policy_distill_weight=2.0,
            policy_teacher_temperature=0.05,
            policy_regret_weight=2.0,
        )

    correct = torch.tensor([0.0, 2.0, 0.5])
    wrong = torch.tensor([0.0, 0.5, 2.0])
    assert loss(correct).item() + 0.1 < loss(wrong).item()


def test_zi_nasc_is_exact_pointwise_identity_at_initialization() -> None:
    torch.manual_seed(11)
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_pooling="scene",
        direct_recovery_value_output="score", direct_recovery_set_context=True,
        direct_recovery_set_context_hidden=16, direct_recovery_set_context_dropout=0.0,
    ).eval()
    x = torch.randn(3, 12)
    group = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        pointwise = model(x, direct_only=True)["direct_recovery_value_logit"]
        setwise = model(x, group_index=group, is_nominal=nominal, direct_only=True)["direct_recovery_value_logit"]
    assert torch.allclose(pointwise, setwise, atol=1.0e-7)


def test_decoupled_ranking_distillation_is_not_corrupted_by_harm_logits() -> None:
    from ocrap.models.losses import direct_uncertainty_recovery_value_loss

    def loss(harm: torch.Tensor) -> torch.Tensor:
        return direct_uncertainty_recovery_value_loss(
            pred_logit=torch.tensor([0.0, 1.5, 0.5]),
            pred_logvar=torch.zeros(3),
            teacher_r_dep=torch.tensor([0.0, 2.0, 1.0]),
            teacher_r_orc=torch.tensor([0.0, 2.0, 1.0]),
            teacher_q=torch.ones((3, 1, 1)), root_probs=torch.ones((3, 1)),
            root_valid=torch.ones((3, 1), dtype=torch.bool),
            option_valid=torch.ones((3, 1), dtype=torch.bool),
            scene_hash=torch.tensor([23, 23, 23]), time_index=torch.tensor([4, 4, 4]),
            macro_type_id=torch.tensor([0, 5, 5]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
            bucket_id=torch.tensor([1, 1, 1]), macro_ids=(5,), bucket_ids=(1,),
            output_mode="score", positive_gain=0.1, negative_gain=0.1,
            point_weight=0.0, centered_weight=0.0, listwise_weight=0.0,
            advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
            opportunity_weight=0.0, harm_weight=0.0, pred_harm_logit=harm,
            setwise_admission_weight=0.0, selective_risk_weight=0.0,
            selective_coverage_weight=0.0, policy_distill_weight=1.0,
            policy_regret_weight=1.0, policy_decouple_admission=True,
            policy_admission_distill_weight=0.0,
        )

    assert torch.allclose(loss(torch.tensor([0.0, -8.0, 8.0])), loss(torch.tensor([0.0, 8.0, -8.0])), atol=1.0e-7)


def test_validation_policy_stats_exposes_worst_regime_regret() -> None:
    from ocrap.cli.train import _finalize_direct_policy_stats
    result = _finalize_direct_policy_stats({
        "regret_sum_all": 0.5, "group_count_all": 4,
        "top1_hit_all": 2, "harmful_switch_all": 1,
        "regret_sum_near": 0.1, "group_count_near": 2,
        "top1_hit_near": 1, "harmful_switch_near": 0,
        "regret_sum_contact": 0.4, "group_count_contact": 2,
        "top1_hit_contact": 1, "harmful_switch_contact": 1,
    })
    assert result["direct_group_regret_mean_near"] == 0.05
    assert result["direct_group_regret_mean_contact"] == 0.2
    assert result["direct_group_regret_mean_worst"] == 0.2
