from __future__ import annotations

import torch

from ocrap.models.ocrap import OCRAPModel, RecoverySetTournament
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("calibrate_policy_risk_v48", Path(__file__).parents[1] / "tools" / "calibrate_policy_risk_v48.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_top1 = _mod._top1


def test_set_tournament_is_permutation_equivariant_and_nominal_pinned() -> None:
    torch.manual_seed(4)
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_output="score",
        direct_recovery_preference_head=False,
        direct_recovery_preference_context=False,
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
    ).eval()
    # Make the score head non-degenerate while preserving shared weights.
    with torch.no_grad():
        model.direct_preference_set_ranker.score.weight.normal_(0.0, 0.1)
    x = torch.randn(4, 12)
    group = torch.zeros((4, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = model(x, group_index=group, is_nominal=nominal, direct_only=True)
    perm = torch.tensor([0, 3, 1, 2])
    out_p = model(x[perm], group_index=group, is_nominal=nominal[perm], direct_only=True)
    inv = torch.argsort(perm)
    assert torch.allclose(out["direct_recovery_rank_logit"], out_p["direct_recovery_rank_logit"][inv], atol=1e-6)
    assert float(out["direct_recovery_rank_logit"][0]) == 0.0


def test_policy_first_no_fallback_abstains_instead_of_selecting_runner_up() -> None:
    groups = [{
        "scene": "s", "time": 1, "fold": 0, "oracle_best_teacher_adv": 0.3,
        "pairs": [
            {"macro": 5, "candidate": 1, "rank_adv": 2.0, "pred_adv": 0.2,
             "opportunity": 0.2, "harm": 0.1, "teacher_adv": 0.3},
            {"macro": 3, "candidate": 2, "rank_adv": 1.0, "pred_adv": 0.1,
             "opportunity": 0.9, "harm": 0.1, "teacher_adv": 0.2},
        ],
    }]
    old = _top1(groups, 0.5, 0.3, {3, 5}, conditional_rank_margin=True, policy_first_no_fallback=False)
    new = _top1(groups, 0.5, 0.3, {3, 5}, conditional_rank_margin=True, policy_first_no_fallback=True)
    assert len(old) == 1 and old[0]["candidate"] == 2
    assert new == []


def test_regime_specific_ordered_evidence_outputs_valid_simplex() -> None:
    model = OCRAPModel(
        input_dim=12, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="mlp", num_layers=1, num_heads=2, dropout=0.0,
        direct_recovery_value_head=True, direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True, direct_recovery_harm_head=True,
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_delta_head=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
    ).eval()
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    p_b = torch.sigmoid(out["direct_recovery_evidence_benefit_logit"])
    p_h = torch.sigmoid(out["direct_recovery_evidence_harm_logit"])
    p_d = 1.0 - p_b - p_h
    assert torch.all(p_d >= -1e-6)
    assert out["direct_recovery_delta_expert_outputs"].shape == (6, 2, 2)


def test_set_tournament_amp_bfloat16_scatter_preserves_output_dtype_and_gradients() -> None:
    """Regression test for mixed-precision indexed scatter in CASTER."""
    torch.manual_seed(48)
    ranker = RecoverySetTournament(
        input_dim=12, hidden_dim=16, num_heads=2, dropout=0.0,
    ).train()
    with torch.no_grad():
        ranker.score.weight.normal_(0.0, 0.1)

    x = torch.randn(6, 12, requires_grad=True)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    # CPU autocast reproduces the same Float32 destination / BFloat16 source
    # condition as CUDA bfloat16 AMP, so this test does not require a GPU.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = ranker(x, groups, nominal)
        loss = scores.square().sum()

    assert scores.dtype == x.dtype
    assert torch.equal(scores[nominal.bool()], torch.zeros(2, dtype=x.dtype))
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0
