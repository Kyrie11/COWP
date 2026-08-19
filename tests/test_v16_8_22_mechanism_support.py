from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cowp.models.losses import candidate_classification_loss, primary_candidate_targets
from cowp.models.set_transport_head import SetTransportCertificateHead


def test_primary_candidate_targets_prefer_explicit_priority_labels():
    batch = {
        "cowp/candidates/valid": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "cowp/candidates/certificate_valid": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        # Global labels intentionally disagree with the protected labels.
        "cowp/candidates/noncoercive_feasible": torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
        "cowp/candidates/false_safe": torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
        "cowp/candidates/priority_eligible": torch.tensor([[1, 1, 0, 1]], dtype=torch.bool),
        "cowp/candidates/priority_noncoercive_feasible": torch.tensor([[1, 0, 0, 1]], dtype=torch.bool),
        "cowp/candidates/priority_false_safe": torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
    }
    ncf, fs, mask, source = primary_candidate_targets(batch)
    assert source == "protected_priority"
    assert mask.tolist() == [[True, True, False, False]]
    assert ncf.bool().tolist() == [[True, False, False, True]]
    assert fs.bool().tolist() == [[False, True, False, False]]

    # Lower score should be supervised as better for the protected NCF candidate,
    # regardless of the deliberately conflicting global labels.
    scores_good = torch.tensor([[-2.0, 2.0, 0.0, 0.0]])
    scores_bad = torch.tensor([[2.0, -2.0, 0.0, 0.0]])
    w = {"candidate_ncf_cls": 1.0, "candidate_false_safe_cls": 1.0}
    assert candidate_classification_loss(scores_good, batch, w)["loss"] < candidate_classification_loss(scores_bad, batch, w)["loss"]


def _transport_inputs(agent2_shift: float = 0.0):
    torch.manual_seed(7)
    B, K, A, M, R, D = 1, 1, 2, 2, 3, 8
    z_agent = torch.randn(B, A, D)
    z_agent[:, 1] += agent2_shift
    z_candidate = torch.randn(B, K, D)
    z_graph = torch.randn(B, D)
    critical_indices = torch.tensor([[0, 1]], dtype=torch.long)
    natural = {
        "mode_latent": torch.randn(B, A, M, D),
        "logits": torch.randn(B, A, M),
        "source_logits": torch.randn(B, A, M, 4),
        "priority_logits": torch.randn(B, A, M),
    }
    response = {
        "safe_logits": torch.randn(B, K, A, R),
        "low_logits": torch.randn(B, K, A, R),
        "valid_logits": torch.full((B, K, A, R), 3.0),
        "mode_logits": torch.randn(B, K, A, R),
        "root_logits": torch.randn(B, K, A, R, M),
        "burden_total": torch.rand(B, K, A, R),
    }
    # Make agent 0 protected and agent 1 ego-priority.
    priority_relation = torch.tensor([[[2, 1]]], dtype=torch.long)
    return dict(
        z_agent=z_agent,
        z_candidate=z_candidate,
        z_graph=z_graph,
        critical_indices=critical_indices,
        natural=natural,
        response=response,
        beta=torch.full((B, A), 0.35),
        critical_mask=torch.ones(B, A, dtype=torch.bool),
        priority_relation=priority_relation,
        audit_relevance_prob=torch.ones(B, K, A),
    )


def test_primary_bcot_has_no_ego_priority_leakage():
    torch.manual_seed(11)
    head = SetTransportCertificateHead(d_model=8, hidden=16, geometry_steps=4)
    head.eval()
    a = _transport_inputs(0.0)
    b = _transport_inputs(50.0)
    # Keep every tensor identical except the embedding of the ego-priority agent.
    for key in ("z_candidate", "z_graph", "critical_indices", "natural", "response", "beta", "critical_mask", "priority_relation", "audit_relevance_prob"):
        b[key] = a[key]
    b["z_agent"] = a["z_agent"].clone()
    b["z_agent"][:, 1] += 50.0
    with torch.no_grad():
        oa = head(**a)
        ob = head(**b)
    assert torch.allclose(oa["candidate_transport_risk"], ob["candidate_transport_risk"], atol=1e-7, rtol=0.0)
    assert torch.allclose(oa["candidate_priority_max_deficit"], ob["candidate_priority_max_deficit"], atol=1e-7, rtol=0.0)
    assert torch.isfinite(oa["candidate_global_transport_risk"]).all()
    assert torch.isfinite(ob["candidate_global_transport_risk"]).all()




def test_primary_bcot_is_vacuously_zero_without_protected_pairs():
    torch.manual_seed(13)
    head = SetTransportCertificateHead(d_model=8, hidden=16, geometry_steps=4)
    head.eval()
    x = _transport_inputs(0.0)
    x["priority_relation"] = torch.tensor([[[1, 1]]], dtype=torch.long)
    with torch.no_grad():
        out = head(**x)
    assert torch.equal(out["candidate_transport_risk"], torch.zeros_like(out["candidate_transport_risk"]))
    assert torch.all(out["candidate_transport_logit"] == -20.0)
    assert torch.isfinite(out["candidate_global_transport_risk"]).all()


def test_mechanism_contrast_audit_accepts_true_within_root_switches(tmp_path: Path):
    K, A, M = 4, 1, 2
    row = {
        "cowp/candidates/valid": np.ones(K, dtype=np.int8),
        "cowp/candidates/certificate_valid": np.ones(K, dtype=np.int8),
        "cowp/candidates/conventional_safe": np.ones(K, dtype=np.int8),
        "cowp/candidates/priority_eligible": np.ones(K, dtype=np.int8),
        "cowp/candidates/priority_noncoercive_feasible": np.array([1, 0, 1, 0], dtype=np.int8),
        "cowp/candidates/priority_false_safe": np.array([0, 1, 0, 1], dtype=np.int8),
        "cowp/critical/valid": np.ones(A, dtype=np.int8),
        "cowp/critical/mechanism_valid": np.ones(A, dtype=np.int8),
        "cowp/witness/rho": np.full((K, A), 2, dtype=np.int32),
        "cowp/witness/opr": np.array([[0.8], [0.2], [0.7], [0.1]], dtype=np.float32),
        "cowp/transport/transported_opr": np.array([[0.8], [0.2], [0.7], [0.1]], dtype=np.float32),
        "cowp/transport/mode_valid": np.ones((K, A, M), dtype=np.int8),
        "cowp/transport/mode_affected": np.array([[[0, 1]], [[1, 1]], [[0, 0]], [[1, 1]]], dtype=np.int8),
        "cowp/transport/root_low_safe_score": np.array([[[0.9, 0.9]], [[0.8, 0.2]], [[0.9, 0.9]], [[0.2, 0.8]]], dtype=np.float32),
    }
    np.savez(tmp_path / "scene.npz", **{k.replace("/", "__"): v for k, v in row.items()})
    out = tmp_path / "audit.json"
    cmd = [
        sys.executable, "-m", "cowp.scripts.74_audit_mechanism_contrast",
        "--cache-dir", str(tmp_path), "--output", str(out), "--strict",
        "--min-rankable-scenes", "1", "--min-rank-pairs", "1",
        "--min-viability-switch-scenes", "1", "--min-viability-switch-roots", "1",
        "--min-recovery-switch-scenes", "1", "--min-recovery-switch-roots", "1",
        "--min-opr-switch-scenes", "1", "--min-partial-opr-pairs", "1",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    audit = json.loads(out.read_text())
    assert audit["pass"] is True
    assert audit["counts"]["rankable_scenes"] == 1
    assert audit["counts"]["viability_switch_roots"] >= 1
    assert audit["counts"]["recovery_switch_roots"] >= 1
    assert audit["counts"]["opr_switch_scenes"] == 1


def test_train_pilot_hard_definition_can_follow_protected_primary(tmp_path: Path):
    def write(name: str, global_ncf: int, priority_ncf: int):
        K = 1
        row = {
            "scenario/id": np.array(name),
            "cowp/candidates/valid": np.ones(K, dtype=np.int8),
            "cowp/candidates/conventional_safe": np.ones(K, dtype=np.int8),
            "cowp/candidates/certificate_valid": np.ones(K, dtype=np.int8),
            "cowp/candidates/noncoercive_feasible": np.array([global_ncf], dtype=np.int8),
            "cowp/candidates/false_safe": np.array([1-global_ncf], dtype=np.int8),
            "cowp/candidates/priority_eligible": np.ones(K, dtype=np.int8),
            "cowp/candidates/priority_noncoercive_feasible": np.array([priority_ncf], dtype=np.int8),
            "cowp/candidates/priority_false_safe": np.array([1-priority_ncf], dtype=np.int8),
            "cowp/candidates/macro_type": np.zeros(K, dtype=np.int32),
            "cowp/candidates/proposal_source": np.zeros(K, dtype=np.int32),
        }
        np.savez(tmp_path / f"{name}.npz", **{k.replace("/", "__"): v for k, v in row.items()})

    # A is globally hard but protected-feasible; B is the reverse.
    write("A", global_ncf=0, priority_ncf=1)
    write("B", global_ncf=1, priority_ncf=0)
    hard = tmp_path / "hard.txt"
    report = tmp_path / "ceiling.json"
    cmd = [
        sys.executable, "-m", "cowp.scripts.45_diagnose_proposal_ceiling",
        "--cache-dir", str(tmp_path), "--output", str(report),
        "--hard-scene-ids", str(hard), "--hard-count", "10",
        "--hard-definition", "protected", "--control-count", "0", "--random-count", "0",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert hard.read_text().split() == ["B"]
    assert json.loads(report.read_text())["hard_definition"] == "protected"
