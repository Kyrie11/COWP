from __future__ import annotations

import torch

from cowp.models.losses import paper_aligned_supervision_batch


def _batch(*, relevant: bool, conflict: bool, affected: bool, recovery: float, root_burden: float):
    return {
        "cowp/candidates/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones(1, 1, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/audit/pair_relevant": torch.tensor([[[relevant]]]),
        "cowp/audit/relevance_mass": torch.tensor([[[1.0 if relevant else 0.0]]]),
        "cowp/natural/weight": torch.ones(1, 1, 1),
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "cowp/natural/beta": torch.full((1, 1), 0.65),
        "cowp/transport/mode_valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/mode_conflict": torch.tensor([[[[conflict]]]]),
        "cowp/transport/mode_affected": torch.tensor([[[[affected]]]]),
        "cowp/transport/mode_retained_low_safe": torch.tensor([[[[not affected]]]]),
        "cowp/transport/response_root_index": torch.zeros(1, 1, 1, 1, dtype=torch.long),
        "cowp/transport/root_low_safe_score": torch.full((1, 1, 1, 1), recovery),
        "cowp/transport/root_min_safe_burden": torch.full((1, 1, 1, 1), root_burden),
        "cowp/response/valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_safe": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_low_burden": torch.tensor([[[[root_burden <= 0.65]]]]),
        "cowp/response/burden_total": torch.full((1, 1, 1, 1), root_burden),
        "cowp/witness/exists": torch.zeros(1, 1, 1, dtype=torch.bool),
    }


def test_irrelevant_global_critical_pair_is_vacuously_noncoercive():
    out = paper_aligned_supervision_batch(
        _batch(relevant=False, conflict=True, affected=True, recovery=0.0, root_burden=1.2), {}
    )
    assert out["cowp/witness/opr"].item() == 1.0
    assert not out["cowp/witness/exists"].item()
    assert out["cowp/witness/pair_noncoercive_feasible"].item()
    assert out["cowp/candidates/noncoercive_feasible"].item()
    assert not out["cowp/candidates/false_safe"].item()


def test_burden_only_affected_root_is_transport_and_witness_support():
    out = paper_aligned_supervision_batch(
        _batch(relevant=True, conflict=False, affected=True, recovery=0.0, root_burden=1.2), {}
    )
    assert out["cowp/witness/natural_conflict_mass"].item() == 0.0
    assert out["cowp/witness/causal_relevance_mass"].item() == 1.0
    assert out["cowp/witness/opr"].item() == 0.0
    assert out["cowp/witness/tail_burden_excess"].item() > 0.5
    assert out["cowp/witness/exists"].item()
    assert not out["cowp/witness/pair_noncoercive_feasible"].item()
    assert not out["cowp/candidates/noncoercive_feasible"].item()
    assert out["cowp/candidates/false_safe"].item()


def test_conflict_only_transport_ablation_collapses_affected_to_conflict():
    import torch
    from cowp.models.set_transport_head import SetTransportCertificateHead

    torch.manual_seed(7)
    head = SetTransportCertificateHead(d_model=16, hidden=8, geometry_steps=4, use_affected_root_transport=False)
    B, K, N, A, M, R, T = 1, 2, 4, 2, 3, 2, 8
    z_candidate = torch.randn(B, K, 16)
    z_agent = torch.randn(B, N, 16)
    z_graph = torch.randn(B, 16)
    critical_indices = torch.tensor([[1, 3]], dtype=torch.long)
    natural = {
        "mode_latent": torch.randn(B, A, M, 16),
        "logits": torch.randn(B, A, M),
        "source_logits": torch.randn(B, A, M, 4),
        "priority_logits": torch.randn(B, A, M),
    }
    response = {
        "mode_latent": torch.randn(B, K, A, R, 16),
        "logits": torch.randn(B, K, A, R),
        "mode_logits": torch.randn(B, K, A, R),
        "valid_logits": torch.randn(B, K, A, R),
        "safe_logits": torch.randn(B, K, A, R),
        "low_logits": torch.randn(B, K, A, R),
        "root_logits": torch.randn(B, K, A, R, M),
        "burden_total": torch.rand(B, K, A, R),
    }
    candidate_traj = torch.randn(B, K, T, 7)
    natural_traj = torch.randn(B, A, M, T, 7)
    beta = torch.full((B, A), 0.65)
    out = head(
        z_agent=z_agent, z_candidate=z_candidate, z_graph=z_graph,
        critical_indices=critical_indices, natural=natural, response=response, beta=beta,
        candidate_traj=candidate_traj, natural_traj=natural_traj,
        critical_mask=torch.ones(B, A, dtype=torch.bool),
    )
    assert torch.allclose(out["mode_affected_prob"], out["mode_conflict_prob"], atol=1e-6, rtol=1e-6)

def test_model_config_exposes_real_v1689_ablation_switches():
    import yaml
    from pathlib import Path
    main = yaml.safe_load(Path("configs/model_cowp_v16_8.yaml").read_text())
    no_rel = yaml.safe_load(Path("configs/model_cowp_v16_8_9_no_causal_relevance.yaml").read_text())
    conflict = yaml.safe_load(Path("configs/model_cowp_v16_8_9_conflict_only_transport.yaml").read_text())
    assert main["ablation"]["use_causal_audit_relevance"] is True
    assert main["ablation"]["use_affected_root_transport"] is True
    assert no_rel["ablation"]["use_causal_audit_relevance"] is False
    assert conflict["ablation"]["use_affected_root_transport"] is False
