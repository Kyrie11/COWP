from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import (
    FactorizedObservationConditionedActionFrontierBridge,
    OCRAPModel,
)

ROOT = Path(__file__).resolve().parents[1]


def _small_model(*, factorized: bool = True, rank_skip: bool = True) -> OCRAPModel:
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
        direct_recovery_evidence_factorized_harm_interaction=factorized,
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


def test_factorized_ocaf_zero_action_exact_zero_for_every_branch() -> None:
    bridge = FactorizedObservationConditionedActionFrontierBridge(
        7, 11, 16, 0.0, component_count=5
    ).eval()
    benefit, harm = bridge(torch.zeros(6, 7), torch.randn(6, 11))
    assert benefit.shape == (6, 16)
    assert harm.shape == (6, 5, 16)
    assert torch.equal(benefit, torch.zeros_like(benefit))
    assert torch.equal(harm, torch.zeros_like(harm))


def test_factorized_ocaf_gradients_are_task_and_component_decoupled() -> None:
    torch.manual_seed(4841)
    bridge = FactorizedObservationConditionedActionFrontierBridge(
        7, 11, 16, 0.0, component_count=3
    )
    action = torch.randn(8, 7)
    observation = torch.randn(8, 11)

    benefit, _ = bridge(action, observation)
    benefit.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.benefit.parameters())
    for branch in bridge.harm_components:
        assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in branch.parameters())

    bridge.zero_grad(set_to_none=True)
    _, harm = bridge(action, observation)
    harm[:, 1].square().mean().backward()
    assert all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in bridge.benefit.parameters())
    for idx, branch in enumerate(bridge.harm_components):
        grads = [p.grad for p in branch.parameters() if p.requires_grad]
        if idx == 1:
            assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)
        else:
            assert all(g is None or torch.equal(g, torch.zeros_like(g)) for g in grads)


def test_full_fcfr_model_exposes_component_diagnostics_and_positive_rank_skip() -> None:
    torch.manual_seed(4841)
    model = _small_model().eval()
    assert isinstance(model.direct_evidence_concord_harm_calibrator, nn.ModuleList)
    assert len(model.direct_evidence_concord_harm_calibrator) == 5
    assert model.direct_evidence_rank_benefit_log_gain is not None
    assert float(torch.nn.functional.softplus(model.direct_evidence_rank_benefit_log_gain).detach()) > 0.0

    x = torch.randn(6, FlatFeatureLayout().total_dim)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        out = model(
            x,
            bucket_id=torch.ones(6, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )

    contexts = out["direct_recovery_evidence_component_harm_interaction_contexts"]
    assert contexts.shape == (6, 5, 16)
    assert torch.equal(contexts[nominal.bool()], torch.zeros_like(contexts[nominal.bool()]))
    # Global support contract is [1,1,1,0,0]; unsupported branches are exact
    # zero placeholders, avoiding two unnecessary OCAF forwards with no semantic change.
    assert torch.equal(contexts[:, 3:], torch.zeros_like(contexts[:, 3:]))
    assert out["direct_recovery_evidence_component_harm_probabilities"].shape == (6, 5)
    assert out["direct_recovery_evidence_predicted_component_margins"].shape == (6, 5)

    # At initialization the concord benefit calibrator's final projection is
    # exactly zero, so the only benefit_raw correction is the positive-gain
    # rank skip.  This makes the monotonicity contract directly testable.
    rank_adv = out["direct_recovery_policy_features"][:, 0]
    gain = out["direct_recovery_evidence_rank_benefit_gain"]
    benefit_raw = out["direct_recovery_evidence_concord_benefit_raw"]
    assert torch.allclose(benefit_raw, gain * rank_adv, atol=1e-6, rtol=1e-6)
    assert torch.all(gain > 0)


def test_fcfr_wrappers_preregister_clean_two_by_two_without_regime_routing() -> None:
    main = (ROOT / "scripts/run_v48_41_fcfr_dedicated.sh").read_text()
    arm = (ROOT / "scripts/run_v48_41_fcfr_ablation_arm.sh").read_text()
    assert "EVIDENCE_DUAL_INTERACTION_BRIDGE=true" in main
    assert "EVIDENCE_FACTORIZED_HARM_INTERACTION=true" in main
    assert "EVIDENCE_RANK_BENEFIT_SKIP=true" in main
    assert "FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw" in main
    assert "EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false" in main
    assert "EVIDENCE_UNBOUNDED_HARM_FACTORS=false" in main
    assert "EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve" in main
    assert "PROPOSAL_TOP_K=5" in main
    for token in ("A)", "B)", "C)", "D)"):
        assert token in arm
    assert "EVIDENCE_FACTORIZED_HARM_INTERACTION=true" in arm
    assert "EVIDENCE_RANK_BENEFIT_SKIP=true" in arm
    # Physical factors are shared continuously; no case-wise policy branch is introduced.
    assert "REGIME_CONDITIONING=true" not in main


def test_v4841_new_flags_are_in_model_and_training_contracts() -> None:
    model_contract = (ROOT / "tools/check_v48_36_ocaf_model_contract.py").read_text()
    training_contract = (ROOT / "tools/check_v48_36_ocaf_training_contract.py").read_text()
    cache = (ROOT / "scripts/adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    for token in ("expect-factorized-harm-interaction", "expect-rank-benefit-skip", "expect-rank-benefit-gain-init"):
        assert token in model_contract
        assert token in training_contract
    for token in ("factorized_harm_interaction", "rank_benefit_skip", "rank_benefit_gain_init"):
        assert token in cache


def test_component_diagnostics_are_propagated_to_calibration_rows() -> None:
    inference = (ROOT / "src/ocrap/models/inference.py").read_text()
    calibration = (ROOT / "tools/calibrate_policy_risk_v48.py").read_text()
    model = (ROOT / "src/ocrap/models/ocrap.py").read_text()
    assert "direct_recovery_evidence_component_harm_probabilities" in model
    assert "direct_recovery_component_harm" in inference
    assert "direct_recovery_component_margins" in inference
    assert "predicted_component_harm" in calibration
    assert "predicted_component_margins" in calibration
