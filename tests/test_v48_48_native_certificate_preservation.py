from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, native: bool) -> OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=3,
        num_options=4,
        d_model=12,
        d_obs=6,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=3,
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
        direct_recovery_evidence_roct_benefit=True,
        direct_recovery_evidence_roct_deployability=True,
        direct_recovery_evidence_roct_scale=3.0,
        direct_recovery_evidence_roct_alpha=0.2,
        direct_recovery_evidence_roct_beta=0.2,
        direct_recovery_evidence_roct_top_m=8,
        direct_recovery_evidence_roct_option_temperature=0.35,
        direct_recovery_evidence_native_certificate_preservation=native,
        direct_recovery_evidence_native_drs_tolerance=0.05,
        direct_recovery_evidence_native_deployability_tolerance=0.05,
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


def test_native_margin_mapping_matches_teacher_sign_convention() -> None:
    m = _model(native=True)
    native = torch.tensor([
        [0.70, 0.60],  # group-0 nominal
        [0.85, 0.75],  # safer recovery -> negative harmful margin
        [0.40, 0.55],  # less DRS, slightly less dep -> positive/zero-ish veto
        [0.20, 0.30],  # group-1 nominal
        [0.35, 0.50],  # safer recovery
    ])
    groups = torch.tensor([[0], [0], [0], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    logits, margins = m._native_certificate_component_logits(native, groups, nominal)
    assert logits is not None and margins is not None
    expected = torch.tensor([
        [-0.05, -0.05],
        [-0.20, -0.20],
        [ 0.25,  0.00],
        [-0.05, -0.05],
        [-0.20, -0.25],
    ])
    assert torch.allclose(margins, expected, atol=1e-6)
    assert torch.allclose(logits, expected / 0.025, atol=1e-6)


def test_native_coupling_overwrites_only_drs_and_deployability_components() -> None:
    torch.manual_seed(4848)
    base = _model(native=False).eval()
    torch.manual_seed(4848)
    native = _model(native=True).eval()
    missing, unexpected = native.load_state_dict(base.state_dict(), strict=True)
    assert not missing and not unexpected  # NCP adds no learned parameters.

    x = torch.randn(8, FlatFeatureLayout().total_dim)
    groups = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    rv = torch.ones((8, 3), dtype=torch.bool)
    ov = torch.ones((8, 4), dtype=torch.bool)
    with torch.no_grad():
        out_base = base(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups,
                        is_nominal=nominal, direct_only=True, root_valid=rv, option_valid=ov)
        out_native = native(x, bucket_id=torch.ones(8, dtype=torch.long), group_index=groups,
                            is_nominal=nominal, direct_only=True, root_valid=rv, option_valid=ov)

    assert "direct_recovery_evidence_native_certificate" in out_native
    assert "direct_recovery_evidence_native_component_margins" in out_native
    got = out_native["direct_recovery_evidence_component_harm_logits"]
    expected_native = out_native["direct_recovery_evidence_native_component_logits"]
    # Nominal rows are intentionally pinned to zero in the published evidence API.
    mask = nominal < 0.5
    assert torch.allclose(got[mask, :2], expected_native[mask], atol=1e-6)
    # Gap and unsupported downstream coordinates are not replaced by NCP.
    assert torch.equal(got[:, 2:], out_base["direct_recovery_evidence_component_harm_logits"][:, 2:])


def test_native_preservation_is_regime_agnostic_and_parameter_free() -> None:
    text = (ROOT / "src/ocrap/models/ocrap.py").read_text(encoding="utf-8")
    block = text[text.index("def _native_certificate_component_logits"):text.index("def _direct_outputs")]
    assert "bucket_id" not in block
    # No regime identifier or bucket enters the computation.  The word
    # "regime-agnostic" may legitimately appear in its explanatory docstring.
    executable = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    assert "bucket" not in executable.lower()
    assert "safe_" not in executable.lower() and "near_" not in executable.lower() and "contact_" not in executable.lower()
    base = _model(native=False)
    ncp = _model(native=True)
    assert set(base.state_dict()) == set(ncp.state_dict())


def test_native_drs_matches_exact_observation_class_predicted_drs() -> None:
    from ocrap.algorithms.ocmero import torch_oc_mero

    root_logits = torch.tensor([[1.2, 0.4, -0.7]], dtype=torch.float32)
    obs_embeddings = torch.tensor(
        [[[0.0, 0.0], [0.05, 0.0], [1.5, 1.5]]], dtype=torch.float32
    )
    margins = torch.tensor(
        [[[0.8, -0.5], [0.7, -0.4], [-0.3, 0.9]]], dtype=torch.float32
    )
    rv = torch.ones((1, 3), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    _sig, native = OCRAPModel._recovery_option_compatibility_signature(
        root_logits,
        obs_embeddings,
        margins,
        tau_obs=0.25,
        alpha=0.2,
        beta=0.2,
        top_m=8,
        option_temperature=0.35,
        root_valid=rv,
        option_valid=ov,
        return_native_certificate=True,
    )

    p = torch.softmax(root_logits, dim=-1)
    diff = obs_embeddings.unsqueeze(2) - obs_embeddings.unsqueeze(1)
    compat = torch.exp(-diff.square().mean(dim=-1) / 0.25)
    eye = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    compat = torch.where(eye, torch.ones_like(compat), compat)
    _rd, _ro, _gap, q = torch_oc_mero(
        margins,
        p,
        compat,
        alpha=0.2,
        beta=0.2,
        option_valid=ov,
        root_valid=rv,
        use_lcvar=True,
        use_obs_kernel=True,
        top_m=8,
    )
    expected = (p * (q.amax(dim=-1) >= 0.0).float()).sum(dim=-1)
    assert torch.allclose(native[:, 0], expected, atol=2e-7, rtol=0.0)
