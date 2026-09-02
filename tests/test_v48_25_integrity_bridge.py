from __future__ import annotations

import ast
from pathlib import Path

import torch

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.ocrap import OCRAPModel


def _small_model(*, bounded: bool) -> OCRAPModel:
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
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.75,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_scale=2.0,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=1.0,
        direct_recovery_evidence_admission_bounded=bounded,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    )


def test_train_cli_forwards_frontier_prior_and_admission_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src" / "ocrap" / "cli" / "train.py").read_text())
    constructor_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OCRAPModel"
    ]
    assert constructor_calls
    keys = {kw.arg for kw in constructor_calls[-1].keywords}
    assert "direct_recovery_evidence_frontier" in keys
    assert "direct_recovery_evidence_component_prior_logit" in keys
    assert "direct_recovery_evidence_admission_bounded" in keys


def test_unbounded_admission_residual_can_cross_bounded_ceiling() -> None:
    bounded = _small_model(bounded=True).eval()
    unbounded = _small_model(bounded=False).eval()
    unbounded.load_state_dict(bounded.state_dict())
    for model in (bounded, unbounded):
        assert model.direct_evidence_concord_admission_calibrator is not None
        with torch.no_grad():
            for param in model.direct_evidence_concord_admission_calibrator.parameters():
                param.zero_()
            model.direct_evidence_concord_admission_calibrator[-1].bias.fill_(3.0)
    torch.manual_seed(4825)
    x = torch.randn(3, 12)
    groups = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    with torch.no_grad():
        b = bounded(x, bucket_id=torch.ones(3, dtype=torch.long), group_index=groups,
                    is_nominal=nominal, direct_only=True)
        u = unbounded(x, bucket_id=torch.ones(3, dtype=torch.long), group_index=groups,
                      is_nominal=nominal, direct_only=True)
    assert float(b["direct_recovery_evidence_calibrator_residual"][:, -1].abs().max()) < 1.01
    assert float(u["direct_recovery_evidence_calibrator_residual"][:, -1].abs().max()) > 2.9


def _stats(*, near_recall: float, contact_recall: float, near_admission: float, contact_admission: float):
    stats: dict[str, float] = {}
    for regime, recall, admission in (
        ("near", near_recall, near_admission),
        ("contact", contact_recall, contact_admission),
    ):
        stats[f"group_count_{regime}"] = 10.0
        stats[f"soft_safe_nll_sum_{regime}"] = 1.0
        stats[f"soft_safe_group_{regime}"] = 2.0
        stats[f"soft_safe_recall_sum_{regime}"] = 1.0
        stats[f"soft_false_admission_sum_{regime}"] = 0.2
        stats[f"soft_harmful_mass_sum_{regime}"] = 0.2
        stats[f"soft_frontier_harmful_mass_sum_{regime}"] = 0.1
        stats[f"soft_safe_mass_sum_{regime}"] = 1.0
        stats[f"soft_safe_regret_sum_{regime}"] = 0.1
        stats[f"positive_count_{regime}"] = 2.0
        stats[f"positive_admission_hit_{regime}"] = recall * 2.0
        stats[f"admission_count_{regime}"] = admission * 10.0
    return stats


def test_integrity_checkpoint_barrier_rejects_all_abstain() -> None:
    cfg = {
        "direct_policy_metric_integrity_min_recall": 0.20,
        "direct_policy_metric_integrity_recall_weight": 20.0,
        "direct_policy_metric_integrity_all_abstain_weight": 8.0,
    }
    abstain = _finalize_direct_policy_stats(
        _stats(near_recall=0.0, contact_recall=0.0, near_admission=0.0, contact_admission=0.0), cfg
    )
    active = _finalize_direct_policy_stats(
        _stats(near_recall=0.25, contact_recall=0.25, near_admission=0.1, contact_admission=0.1), cfg
    )
    assert abstain["direct_integrity_all_abstain"] == 1.0
    assert active["direct_integrity_all_abstain"] == 0.0
    assert abstain["direct_integrity_selection_risk"] > active["direct_integrity_selection_risk"]


def test_certificate_and_shadow_contracts_separate_engineering_failure_from_gate_rejection() -> None:
    root = Path(__file__).resolve().parents[1]
    calibrator = (root / "tools" / "calibrate_policy_risk_v48.py").read_text()
    certificate = (root / "scripts" / "calibrate_v48_25_certificate_pool.sh").read_text()
    runtime = (root / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    assert "--development-fit-only" in calibrator
    assert "--verification-only" in calibrator
    assert 'return 3  # protocol/artifact failure' not in calibrator
    assert "Natural-gate rejection" in calibrator
    assert "--frozen-rule-json" in certificate
    assert "certificate_labels_used_for_threshold_fit" in certificate
    assert "adaptation_dev_frozen_rule_diagnostic_only" in runtime


def test_v48_25_uses_dev_index_and_balances_ablations_across_two_a30s() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = (root / "scripts" / "run_v48_25_integrity_dedicated.sh").read_text()
    parallel = (root / "scripts" / "run_v48_25_parallel_ablations.sh").read_text()
    assert "VAL_GROUP_INDEX" in controller
    assert "build_adapt_dev_teacher_index.log" in controller
    assert "VAL_GROUP_INDEX=\"$VAL_GROUP_INDEX\"" in controller
    assert "max_concurrent_tasks':2" in parallel
    assert "one task per A30" in parallel
    assert "A_wiring_fix_bounded" in parallel
    assert "D_full_integrity_bridge" in parallel
    assert (root / "tools" / "check_v48_25_regime_targets.py").is_file()
