from __future__ import annotations

import json
from pathlib import Path

import torch

from ocrap.models.losses import (
    observation_consistent_frontier_calibration_loss,
    recovery_conflict_pair_weights,
)
from ocrap.cli.train import _obs_bce


def test_recovery_conflict_weights_emphasize_incompatible_recovery_pair() -> None:
    # Root 0 can only recover with option 0, root 1 only with option 1.
    m = torch.tensor([[[2.0, -2.0], [-2.0, 2.0]]])
    rv = torch.tensor([[True, True]])
    ov = torch.tensor([[True, True]])
    w = recovery_conflict_pair_weights(m, rv, ov, temperature=0.10, conflict_scale=3.0, max_weight=4.0)
    assert w.shape == (1, 2, 2)
    assert float(w[0, 0, 1]) > 3.5
    assert float(w[0, 0, 0]) < float(w[0, 0, 1])


def test_decision_weighted_obs_changes_gradient_allocation_not_label() -> None:
    pred = torch.tensor([[[1.0, 0.80, 0.20], [0.80, 1.0, 0.40], [0.20, 0.40, 1.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    rv = torch.tensor([[True, True, True]])
    pair = torch.ones_like(pred)
    pair[:, 0, 1] = 4.0; pair[:, 1, 0] = 4.0
    loss = _obs_bce(pred, target, rv, balanced=False, pair_weights=pair)
    loss.backward()
    assert pred.grad is not None
    # Same BCE label, but the weighted pair receives a larger magnitude gradient.
    assert abs(float(pred.grad[0, 0, 1])) > abs(float(pred.grad[0, 0, 2]))


def _frontier_fixture(correct: bool):
    # One scene-time group: row 0 nominal, row 1 recovery candidate.
    teacher_r_dep = torch.tensor([-0.5, 1.0])  # candidate has much better deployability
    teacher_q = torch.tensor([
        [[-0.5, -0.5], [-0.5, -0.5]],
        [[ 1.0, -1.0], [-1.0,  1.0]],
    ])
    if correct:
        pred_r_dep = teacher_r_dep.clone().requires_grad_(True)
        pred_q = teacher_q.clone().requires_grad_(True)
    else:
        # Wrong frontier: predict nominal as better and candidate classes unrecoverable.
        pred_r_dep = torch.tensor([1.0, -0.5], requires_grad=True)
        pred_q = torch.tensor([
            [[1.0, 1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ], requires_grad=True)
    probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    rv = torch.ones((2, 2), dtype=torch.bool)
    ov = torch.ones((2, 2), dtype=torch.bool)
    sh = torch.tensor([11, 11]); ti = torch.tensor([7, 7]); nominal = torch.tensor([1.0, 0.0])
    return pred_r_dep, pred_q, teacher_r_dep, teacher_q, probs, rv, ov, sh, ti, nominal


def test_recovery_frontier_loss_prefers_correct_candidate_relative_geometry() -> None:
    good = _frontier_fixture(True); bad = _frontier_fixture(False)
    lg = observation_consistent_frontier_calibration_loss(*good)
    lb = observation_consistent_frontier_calibration_loss(*bad)
    assert torch.isfinite(lg) and torch.isfinite(lb)
    assert float(lg.detach()) < float(lb.detach())
    lb.backward()
    assert bad[0].grad is not None and float(bad[0].grad.abs().sum()) > 0
    assert bad[1].grad is not None and float(bad[1].grad.abs().sum()) > 0


def test_v4847_arm_matrix_no_regime_policy_and_two_gpu_pairing() -> None:
    repo = Path(__file__).resolve().parents[1]
    arm = (repo / 'scripts/run_v48_47_dsofr_ablation_arm.sh').read_text()
    launcher = (repo / 'scripts/run_v48_47_dsofr_2x2_two_gpu.sh').read_text()
    witness = (repo / 'scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh').read_text()
    assert 'TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class' in arm
    assert 'EVAL_OPTION_EXECUTION_SEMANTICS=observation_class' in arm
    assert 'export V4847_DECISION_OBS=1' in arm
    assert 'export V4847_RECOVERY_FRONTIER=1' in arm
    assert "strategy_regime_conditioning':False" in arm
    assert 'for pair in "A B" "C D"' in launcher
    assert 'run_arm "$left" "$GPU0"' in launcher and 'run_arm "$right" "$GPU1"' in launcher
    assert 'prefixes="obs_embed_head"' in witness and 'prefixes="margin_head"' in witness
    assert 'root_logit_head' not in witness.split('case "$STAGE" in',1)[1].split('[[ -f "$INIT_CKPT"',1)[0]


def test_v4847_d_reuse_is_fail_closed_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / 'tools/reuse_v48_47_witness_stage.py').read_text()
    for token in ('source_sha','checkpoint_sha','train_mix','val_mix','group_sha','obs_conflict_scale','obs_conflict_temperature'):
        assert token in text
    adapter = (repo / 'scripts/adapt_ocrap_v48_36_ocaf_variant.sh').read_text()
    assert 'reuse_v48_47_witness_stage.py' in adapter
    assert 'V4847_OBS_REUSE_BASE' in adapter


def test_v4847_witness_only_fast_path_preserves_active_outputs_and_rng() -> None:
    from ocrap.models.ocrap import OCRAPModel
    from ocrap.cli.train import _keep_fully_frozen_modules_in_eval

    model = OCRAPModel(
        input_dim=16, num_roots=2, num_options=3, d_model=16, d_obs=8,
        encoder_type='mlp', dropout=0.2,
        direct_recovery_value_head=True,
        direct_recovery_set_tournament=True,
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_delta_mode='ordinal_evidence',
    )
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('obs_embed_head'))
    model.train()
    _keep_fully_frozen_modules_in_eval(model)
    assert model.obs_embed_head.training

    x = torch.randn(4, 16)
    group_index = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 2, 2], [0, 2, 2]])
    is_nominal = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    root_valid = torch.ones(4, 2, dtype=torch.bool)
    option_valid = torch.ones(4, 3, dtype=torch.bool)

    torch.manual_seed(48147)
    full = model(
        x, group_index=group_index, is_nominal=is_nominal,
        root_valid=root_valid, option_valid=option_valid,
    )
    full_rng = torch.random.get_rng_state().clone()

    # DWOK does not consume predicted margins; skip the KxL margin head too.
    torch.manual_seed(48147)
    fast_obs = model(
        x, group_index=group_index, is_nominal=is_nominal,
        root_valid=root_valid, option_valid=option_valid, witness_only=True,
        witness_observation_only=True,
    )
    fast_obs_rng = torch.random.get_rng_state().clone()
    assert set(fast_obs) == {'root_logits', 'obs_embeddings', 'c_star'}
    for key in fast_obs:
        torch.testing.assert_close(fast_obs[key], full[key], rtol=0.0, atol=0.0)
    assert torch.equal(full_rng, fast_obs_rng)

    # DRFC needs margins, but still skips every downstream/direct frozen head.
    torch.manual_seed(48147)
    fast_frontier = model(
        x, group_index=group_index, is_nominal=is_nominal,
        root_valid=root_valid, option_valid=option_valid, witness_only=True,
    )
    fast_frontier_rng = torch.random.get_rng_state().clone()
    assert set(fast_frontier) == {'root_logits', 'margins', 'obs_embeddings', 'c_star'}
    for key in fast_frontier:
        torch.testing.assert_close(fast_frontier[key], full[key], rtol=0.0, atol=0.0)
    # All skipped submodules are fully frozen and forced to eval, so omitting
    # them must not perturb the stochastic trajectory seen by the active head.
    assert torch.equal(full_rng, fast_frontier_rng)


def test_v4847_witness_stage_enables_exact_fast_path_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[1]
    trainer = (repo / 'scripts/train_ocrap_v48_trac_sr.sh').read_text()
    witness = (repo / 'scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh').read_text()
    reuse = (repo / 'tools/reuse_v48_47_witness_stage.py').read_text()
    assert 'training.witness_fast_path="${WITNESS_FAST_PATH:-}"' in trainer
    assert 'training.frozen_modules_eval="${FROZEN_MODULES_EVAL:-false}"' in trainer
    assert 'WITNESS_FAST_PATH="$STAGE" FROZEN_MODULES_EVAL=true' in witness
    assert "'witness_fast_path':stage,'frozen_modules_eval':True" in witness
    assert "c.get('witness_fast_path')==args.expected_stage" in reuse
    assert "c.get('frozen_modules_eval') is True" in reuse
