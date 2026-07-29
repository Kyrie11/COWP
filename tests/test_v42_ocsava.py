from dataclasses import asdict

import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select


def test_certificate_action_adapter_accepts_raw_candidate_path():
    layout = FlatFeatureLayout(feature_max_agents=2)
    model = OCRAPModel(
        input_dim=layout.total_dim,
        num_roots=2,
        num_options=3,
        d_model=16,
        d_obs=8,
        encoder_type="structured_transformer",
        feature_layout=asdict(layout),
        num_layers=1,
        num_heads=2,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_value_output="score",
    )
    out = model(torch.randn(4, layout.total_dim))
    assert out["direct_recovery_value_logit"].shape == (4,)
    assert model.direct_candidate_raw_dim > 0
    assert model.direct_recovery_value_output == "score"


def _score_loss(logits):
    return direct_uncertainty_recovery_value_loss(
        torch.tensor(logits, dtype=torch.float32, requires_grad=True),
        torch.tensor([-4.0, -4.0, -4.0], dtype=torch.float32, requires_grad=True),
        torch.tensor([-2.0, 0.6, -2.1]),
        torch.tensor([-1.8, 0.7, -1.9]),
        torch.tensor([[[-2.0]], [[2.0]], [[-2.2]]]),
        torch.ones((3, 1)),
        torch.ones((3, 1), dtype=torch.bool),
        torch.ones((3, 1), dtype=torch.bool),
        torch.tensor([17, 17, 17]),
        torch.tensor([5, 5, 5]),
        torch.tensor([0, 2, 3]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1, 1, 1]),
        output_mode="score",
        point_weight=0.0,
        pairwise_weight=1.5,
        top_rank_weight=1.0,
        success_temperature=0.1,
    )


def test_unbounded_score_loss_prefers_correct_top1_order():
    aligned = _score_loss([0.0, 0.8, -0.3])
    inverted = _score_loss([0.0, -0.3, 0.8])
    assert aligned.item() < inverted.item()
    aligned.backward()


def _select(*, top1: bool):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.1, 0.1]),
        r_dep=np.array([0.4, 0.4, 0.4]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.ones(3, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.array([0.05, 0.05, 0.05]),
        pred_drs=np.array([0.95, 0.95, 0.95]),
        nominal_deviation=np.array([0.0, 0.02, 0.02]),
        pred_direct_value=np.array([-2.0, -1.0, -1.2]),
        pred_direct_std=np.zeros(3),
        candidate_macro_names=["nominal", "brake", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="brake,yield",
        direct_value_uncertainty_mode="none",
        direct_value_min_advantage_lcb=0.05,
        direct_value_min_candidate_value=0.45,  # ignored for score mode
        direct_value_score_mode=True,
        direct_value_top1_only=top1,
        direct_value_challenge_nominal=True,
        direct_value_bonus=1.0,
        stress_rescue_challenge_nominal=True,
        direct_value_max_consecutive=1,
        previous_selected_macro="brake",
        same_macro_run_length=1,
    )


def test_top1_rule_does_not_fall_through_after_stateful_block():
    # The top score is brake.  Once its consecutive-action guard blocks it, the
    # selection-conditional rule abstains instead of silently trying yield,
    # which would not be covered by the top-1 calibration event.
    top1 = _select(top1=True)
    assert top1.selected_index == 0


def test_score_mode_does_not_apply_probability_floor():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.1]),
        r_dep=np.array([0.4, 0.4]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.array([0.05, 0.05]),
        pred_drs=np.array([0.95, 0.95]),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([-2.0, -1.0]),
        pred_direct_std=np.zeros(2),
        candidate_macro_names=["nominal", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="yield",
        direct_value_uncertainty_mode="none",
        direct_value_min_advantage_lcb=0.05,
        direct_value_min_candidate_value=0.45,
        direct_value_score_mode=True,
        direct_value_top1_only=True,
        direct_value_challenge_nominal=True,
        direct_value_bonus=1.0,
        stress_rescue_challenge_nominal=True,
    )
    assert sel.selected_index == 1
    assert "direct_value" in sel.reason


def test_selection_conditional_top1_abstains_when_top_action_is_unadmitted():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.05, 0.05]),
        r_dep=np.array([0.5, -1.0, 0.5]),
        hard=np.zeros(3),
        harm=np.zeros(3),
        feasible=np.ones(3, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.array([0.0, 0.0, 0.0]),
        pred_drs=np.array([1.0, 1.0, 1.0]),
        nominal_deviation=np.array([0.0, 0.02, 0.02]),
        pred_direct_value=np.array([0.0, 2.0, 1.0]),
        pred_direct_std=np.zeros(3),
        candidate_macro_names=["nominal", "brake", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="brake,yield",
        direct_value_uncertainty_mode="none",
        direct_value_min_advantage_lcb=0.05,
        direct_value_score_mode=True,
        direct_value_top1_only=True,
        direct_value_challenge_nominal=True,
        direct_value_bonus=5.0,
        stress_rescue_challenge_nominal=True,
    )
    # Candidate 1 is the calibrated top-1 actionable score but lacks the base
    # OC-MERO admission certificate.  The rule must abstain rather than fall
    # through to candidate 2, because rank 2 is outside the calibrated event.
    assert sel.selected_index == 0
    assert not bool(sel.admitted[1])
