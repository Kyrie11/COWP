from __future__ import annotations

import torch

from cowp.models.losses import symmetric_class_balanced_bce_with_logits
from cowp.models.set_transport_head import SetTransportCertificateHead


def test_symmetric_class_balance_handles_positive_majority() -> None:
    # v16.6 false-safe supervision was about 87% positive.  Symmetric balancing
    # should give the minority negatives comparable aggregate gradient mass.
    logits = torch.zeros(10, requires_grad=True)
    target = torch.tensor([1.0] * 9 + [0.0])
    mask = torch.ones(10, dtype=torch.bool)
    loss = symmetric_class_balanced_bce_with_logits(logits, target, mask)
    loss.backward()
    assert logits.grad is not None
    pos_mass = logits.grad[:9].abs().sum()
    neg_mass = logits.grad[9:].abs().sum()
    assert torch.isclose(pos_mass, neg_mass, rtol=1e-5, atol=1e-6)


def test_candidate_calibrator_is_monotone_in_every_deficit() -> None:
    raw_weight = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    threshold = torch.tensor(0.0)
    log_scale = torch.tensor(1.0)
    base = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    _, base_risk, weight = SetTransportCertificateHead._monotone_calibrate(
        base, raw_weight, threshold, log_scale
    )
    assert torch.all(weight > 0)
    assert torch.isclose(weight.sum(), torch.tensor(1.0), atol=1e-6)
    for j in range(base.shape[-1]):
        raised = base.clone()
        raised[0, j] += 0.2
        _, risk, _ = SetTransportCertificateHead._monotone_calibrate(
            raised, raw_weight, threshold, log_scale
        )
        assert risk.item() > base_risk.item()


def test_root_alignment_respects_structural_source_identity() -> None:
    from cowp.models.losses import _gt_to_pred_natural_assignment

    # Geometrically, predicted root 0 is the closer match to GT root 0, but its
    # source is NEU.  Predicted root 1 is a slightly worse geometric match and is
    # structurally OBS, so same-root transport should choose root 1.
    gt = torch.zeros(1, 1, 1, 10, 5)
    pred_traj = torch.zeros(1, 1, 2, 10, 5)
    pred_traj[:, :, 1, :, 0] = 0.2
    source_logits = torch.full((1, 1, 2, 4), -8.0)
    source_logits[:, :, 0, 1] = 8.0  # NEU
    source_logits[:, :, 1, 0] = 8.0  # OBS
    pred = {
        "_natural_pred_traj": pred_traj,
        "_natural_pred_source_logits": source_logits,
    }
    batch = {
        "cowp/natural/traj": gt,
        "cowp/natural/source": torch.zeros(1, 1, 1, dtype=torch.long),
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
    }
    assignment = _gt_to_pred_natural_assignment(pred, batch)
    assert assignment is not None
    assert assignment.item() == 1


def test_priority_bte_reports_real_upper_tail_cvar_and_ncf_precision() -> None:
    import numpy as np

    from cowp.waymax_eval.rollout import _LearnedMetricsAccumulator

    acc = _LearnedMetricsAccumulator()
    for tail_value in (1.0, 2.0, 3.0, 4.0):
        label = {
            "cowp/candidates/valid": np.asarray([True]),
            "cowp/candidates/noncoercive_feasible": np.asarray([True]),
            "cowp/candidates/false_safe": np.asarray([False]),
            "cowp/candidates/conventional_safe": np.asarray([True]),
            "cowp/candidates/trajectory": np.zeros((1, 2, 3), dtype=np.float32),
            "cowp/critical/valid": np.asarray([True]),
            "cowp/witness/exists": np.asarray([[False]]),
            "cowp/witness/rho": np.asarray([[2]], dtype=np.int64),
            "cowp/witness/tail_burden_excess": np.asarray([[tail_value]], dtype=np.float32),
            "cowp/witness/burden_total": np.asarray([[0.0]], dtype=np.float32),
            "cowp/witness/opr": np.asarray([[1.0]], dtype=np.float32),
            "cowp/witness/min_safe_burden": np.asarray([[0.0]], dtype=np.float32),
            "cowp/natural/beta": np.asarray([0.65], dtype=np.float32),
        }
        acc.add_selection(0, np.asarray([True]), label)

    metrics = acc.finish(auprc=1.0, rank_good=1, rank_total=1, witness_threshold=0.5)
    assert metrics["LearnedAcceptNCFPrecision"] == 1.0
    assert metrics["PriorityCertificate/AcceptNCFPrecision"] == 1.0
    assert metrics["PriorityBurden/MeanWorstRelationBTE"] == 2.5
    assert metrics["PriorityBurden/BTE_CVaR_25"] == 4.0
    assert metrics["PriorityBurden/BTE_CVaR_25_Count"] == 1
