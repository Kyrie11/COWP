from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, partial: bool, rank_skip: bool = False) -> OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_interaction_hidden=16,
        direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=True,
        direct_recovery_evidence_factorized_harm_interaction=False,
        direct_recovery_evidence_partial_pool_harm_residual=partial,
        direct_recovery_evidence_partial_pool_harm_residual_scale=0.50,
        direct_recovery_evidence_rank_benefit_skip=rank_skip,
        direct_recovery_evidence_rank_benefit_gain_init=1.0,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_prior_mode="joint_reserve",
        direct_recovery_evidence_reserve_factor_alignment=True,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
    )


def _batch():
    torch.manual_seed(4842)
    x = torch.randn(6, FlatFeatureLayout().total_dim)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return x, groups, nominal


def test_partial_pool_residual_is_exact_identity_at_initialisation() -> None:
    torch.manual_seed(4842)
    base = _model(partial=False).eval()
    torch.manual_seed(4842)
    hpfr = _model(partial=True).eval()
    # Copy every shared parameter from the baseline; the only unmatched keys are
    # the new zero-output component residual heads.
    missing, unexpected = hpfr.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(
        key.startswith("direct_evidence_concord_harm_component_residuals.")
        for key in missing
    )

    x, groups, nominal = _batch()
    with torch.no_grad():
        a = base(x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
        b = hpfr(x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    assert torch.equal(a["direct_recovery_evidence_component_harm_logits"], b["direct_recovery_evidence_component_harm_logits"])
    residual = b["direct_recovery_evidence_partial_pool_harm_component_residuals"]
    assert torch.equal(residual, torch.zeros_like(residual))


def test_partial_pool_component_residual_gradient_is_specialized_and_detached() -> None:
    torch.manual_seed(4842)
    model = _model(partial=True)
    x, groups, nominal = _batch()
    out = model(x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    residual = out["direct_recovery_evidence_partial_pool_harm_component_residuals"]
    residual[:, 1].sum().backward()

    # Detached residual input must not rotate the shared OCAF bridge.
    harm_bridge = model.direct_evidence_interaction_bridge.harm
    assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in harm_bridge.parameters())

    heads = model.direct_evidence_concord_harm_component_residuals
    assert heads is not None
    for idx, head in enumerate(heads):
        params = [p for p in head.parameters() if p.requires_grad]
        if idx == 1:
            assert params and any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        elif params:
            assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in params)


def test_rank_scalar_is_valid_exact_trainable_contract_key() -> None:
    stage = (ROOT / "scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh").read_text()
    assert "k == prefix or k.startswith(prefix+'.')" in stage
    transfer = (ROOT / "tools/check_v48_36_stage_transfer.py").read_text()
    assert '"direct_evidence_rank_benefit_log_gain"' in transfer
    assert 'key == prefix or key.startswith(prefix + ".")' in transfer


def test_hpfr_wrappers_preregister_clean_two_by_two() -> None:
    main = (ROOT / "scripts/run_v48_42_hpfr_dedicated.sh").read_text()
    arm = (ROOT / "scripts/run_v48_42_hpfr_ablation_arm.sh").read_text()
    assert "EVIDENCE_DUAL_INTERACTION_BRIDGE=true" in main
    assert "EVIDENCE_FACTORIZED_HARM_INTERACTION=false" in main
    assert "EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=true" in main
    assert "EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL_SCALE=0.50" in main
    assert "EVIDENCE_RANK_BENEFIT_SKIP=true" in main
    assert "FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw" in main
    assert "EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false" in main
    assert "EVIDENCE_UNBOUNDED_HARM_FACTORS=false" in main
    for token in ("A)", "B)", "C)", "D)"):
        assert token in arm
    assert "EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=true" in arm
    assert "EVIDENCE_RANK_BENEFIT_SKIP=true" in arm
    assert "REGIME_CONDITIONING=true" not in main


def test_hpfr_flags_are_cache_checkpoint_and_contract_bound() -> None:
    train = (ROOT / "src/ocrap/cli/train.py").read_text()
    inference = (ROOT / "src/ocrap/models/inference.py").read_text()
    cache = (ROOT / "scripts/adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    model_contract = (ROOT / "tools/check_v48_36_ocaf_model_contract.py").read_text()
    training_contract = (ROOT / "tools/check_v48_36_ocaf_training_contract.py").read_text()
    for token in (
        "direct_recovery_evidence_partial_pool_harm_residual",
        "direct_recovery_evidence_partial_pool_harm_residual_scale",
    ):
        assert token in train
        assert token in inference
    assert "partial_pool_harm_residual=" in cache
    assert "partial_pool_harm_residual_scale=" in cache
    assert "expect-partial-pool-harm-residual" in model_contract
    assert "expect-partial-pool-harm-residual" in training_contract


def test_component_teacher_terms_are_emitted_for_post_run_diagnosis() -> None:
    calibration = (ROOT / "tools/calibrate_policy_risk_v48.py").read_text()
    assert "component_veto_terms_numpy" in calibration
    assert '"teacher_component_veto_terms"' in calibration
