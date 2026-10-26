from __future__ import annotations

import torch


def natural_root_alignment_cost(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    *,
    pred_source_logits: torch.Tensor | None = None,
    gt_source: torch.Tensor | None = None,
    source_penalty_m: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-aware multi-horizon cost for unordered natural-root alignment.

    A full-horizon ADE-only match can cross source identities after an
    intervention (for example, an OBS root can be paired with a geometrically
    similar neutral/yield root).  That corrupts the very same-root supervision
    used by transport.  The cost below shares one distance tensor across
    1/3/5/8 s summaries and adds a source NLL measured in metre-equivalent
    units.  It is used only under ``no_grad`` for hard label alignment.

    Returns ``(alignment_cost, full_horizon_ade)`` with shape
    ``[B, A, M_pred, M_gt]``.
    """
    pred = pred_traj.float()
    gt = gt_traj.float()
    if pred.ndim != 5 or gt.ndim != 5:
        raise ValueError(f"Expected [B,A,M,T,D], got {tuple(pred.shape)} and {tuple(gt.shape)}")
    t = min(int(pred.shape[-2]), int(gt.shape[-2]))
    if t <= 0:
        raise ValueError("Natural trajectories must have a non-empty horizon")
    dist = torch.linalg.vector_norm(
        pred[..., :t, :2][:, :, :, None, :, :]
        - gt[..., :t, :2][:, :, None, :, :, :],
        dim=-1,
    )
    cumulative = dist.cumsum(dim=-1)
    # Preserve long-horizon identity while giving early conflict geometry enough
    # weight that roots cannot swap only because their terminal positions cross.
    steps = (min(10, t), min(30, t), min(50, t), t)
    weights = (0.15, 0.25, 0.25, 0.35)
    cost = torch.zeros_like(cumulative[..., 0])
    for step, weight in zip(steps, weights):
        cost = cost + float(weight) * cumulative[..., step - 1] / float(step)
    full_ade = cumulative[..., t - 1] / float(t)

    if (
        pred_source_logits is not None
        and gt_source is not None
        and pred_source_logits.ndim == 4
        and gt_source.ndim == 3
        and pred_source_logits.shape[:3] == pred.shape[:3]
        and gt_source.shape[:3] == gt.shape[:3]
    ):
        prob = torch.softmax(pred_source_logits.float(), dim=-1).clamp_min(1.0e-8)
        source_count = int(prob.shape[-1])
        gt_src = gt_source.long().clamp(0, max(source_count - 1, 0))
        expanded_prob = prob[:, :, :, None, :].expand(
            prob.shape[0], prob.shape[1], prob.shape[2], gt_src.shape[2], source_count
        )
        src_idx = gt_src[:, :, None, :, None].expand(
            gt_src.shape[0], gt_src.shape[1], prob.shape[2], gt_src.shape[2], 1
        )
        match_prob = torch.gather(expanded_prob, dim=-1, index=src_idx).squeeze(-1)
        cost = cost + max(float(source_penalty_m), 0.0) * (-torch.log(match_prob))
    return cost, full_ade
