import numpy as np
import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select


def _direct_loss(logits):
    return direct_uncertainty_recovery_value_loss(
        torch.tensor(logits, dtype=torch.float32, requires_grad=True),
        torch.tensor([-4.0, -4.0], dtype=torch.float32, requires_grad=True),
        torch.tensor([-2.0, 0.5]),
        torch.tensor([-1.8, 0.6]),
        torch.tensor([[[-2.0]], [[2.0]]]),
        torch.ones((2, 1)),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
        torch.tensor([17, 17]),
        torch.tensor([5, 5]),
        torch.tensor([0, 2]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1, 1]),
        success_temperature=0.1,
    )


def test_direct_value_loss_prefers_teacher_aligned_ordering():
    inverted = _direct_loss([2.0, -2.0])
    aligned = _direct_loss([-2.0, 2.0])
    assert aligned.item() < inverted.item()
    inverted.backward()


def test_direct_value_head_outputs_bounded_value_parameters():
    model = OCRAPModel(
        input_dim=12,
        num_roots=2,
        num_options=3,
        d_model=16,
        d_obs=8,
        num_heads=2,
        direct_recovery_value_head=True,
    )
    out = model(torch.randn(4, 12))
    assert out["direct_recovery_value_logit"].shape == (4,)
    assert out["direct_recovery_value_logvar"].shape == (4,)
    assert torch.all(out["direct_recovery_value_logvar"] <= 2.0)
    assert torch.all(out["direct_recovery_value_logvar"] >= -7.0)


def _select(std):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.9]),
        r_dep=np.array([0.3, 0.25]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.array([True, True]),
        gamma_rec=0.0,
        pred_gap=np.array([0.05, 0.08]),
        pred_drs=np.array([0.95, 0.92]),
        pred_direct_value=np.array([0.20, 0.80]),
        pred_direct_std=np.array([std, std]),
        candidate_macro_names=["nominal", "brake"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="brake,yield,merge,stabilize",
        direct_value_lcb_z=1.0,
        direct_value_min_advantage_lcb=0.05,
        direct_value_min_candidate_value=0.45,
        direct_value_max_candidate_std=0.35,
        direct_value_challenge_nominal=True,
        stress_rescue_challenge_nominal=True,
        direct_value_bonus=0.5,
    )


def test_direct_value_lcb_can_challenge_admitted_nominal():
    sel = _select(0.05)
    assert sel.selected_index == 1
    assert sel.reason == "best_direct_value_lcb_guarded_challenge"


def test_direct_value_uncertainty_blocks_unsafe_challenge():
    sel = _select(0.50)
    assert sel.selected_index == 0
    assert sel.reason == "nominal_calibrated_admitted"


def test_scene_time_validation_sampler_is_deterministic():
    from ocrap.cli.train import SceneTimeBatchSampler

    sampler = SceneTimeBatchSampler(
        [[0, 1], [2, 3, 4], [5]],
        batch_size=4,
        replacement=False,
        shuffle_within_group=False,
        shuffle_groups=False,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second == [[0, 1], [2, 3, 4, 5]]


def test_additive_conformal_bound_ignores_self_reported_std_and_challenges():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.9]), r_dep=np.array([0.3, 0.25]),
        hard=np.zeros(2), harm=np.zeros(2), feasible=np.array([True, True]),
        gamma_rec=0.0, pred_gap=np.array([0.05, 0.08]), pred_drs=np.array([0.95, 0.92]),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([0.20, 0.50]), pred_direct_std=np.array([9.0, 9.0]),
        candidate_macro_names=["nominal", "brake"], regime_name="test_near_contact",
        direct_value_certificate=True, direct_value_macro_allowlist="brake",
        direct_value_uncertainty_mode="additive", direct_value_additive_q=0.10,
        direct_value_min_nominal_deviation=0.002, direct_value_min_advantage_lcb=0.05,
        direct_value_min_candidate_value=0.40, direct_value_challenge_nominal=True,
        stress_rescue_challenge_nominal=True, direct_value_bonus=0.5,
    )
    assert sel.selected_index == 1
    assert "direct_value" in sel.reason


def test_actionability_gate_blocks_nominal_equivalent_direct_candidate():
    sel = calibrated_constrained_select(
        utility=np.array([1.0, 0.9]), r_dep=np.array([0.3, 0.25]),
        hard=np.zeros(2), harm=np.zeros(2), feasible=np.array([True, True]),
        gamma_rec=0.0, pred_gap=np.array([0.05, 0.08]), pred_drs=np.array([0.95, 0.92]),
        nominal_deviation=np.array([0.0, 1.0e-5]),
        pred_direct_value=np.array([0.20, 0.80]), pred_direct_std=np.array([0.01, 0.01]),
        candidate_macro_names=["nominal", "brake"], regime_name="test_near_contact",
        direct_value_certificate=True, direct_value_macro_allowlist="brake",
        direct_value_uncertainty_mode="additive", direct_value_additive_q=0.0,
        direct_value_min_nominal_deviation=0.002, direct_value_min_advantage_lcb=0.05,
        direct_value_min_candidate_value=0.40, direct_value_challenge_nominal=True,
        stress_rescue_challenge_nominal=True, direct_value_bonus=0.5,
    )
    assert sel.selected_index == 0


def test_candidate_concat_direct_head_output_shape():
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=16, d_obs=8,
        num_heads=2, direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat",
    )
    out = model(torch.randn(3, 12))
    assert out["direct_recovery_value_logit"].shape == (3,)
