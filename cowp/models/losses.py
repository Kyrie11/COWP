from __future__ import annotations

import torch
import torch.nn.functional as F

from cowp.core.constants import NaturalSource


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean over a boolean mask without leaking padded NaN/Inf values.

    ``value * mask`` is unsafe when invalid padded slots contain NaN because
    NaN * 0 is still NaN.  This helper replaces non-selected and non-finite
    values with zero before reduction.
    """
    mask_b = mask.bool()
    if value.shape != mask_b.shape:
        mask_b = torch.broadcast_to(mask_b, value.shape)
    safe_value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_value = torch.where(mask_b, safe_value, torch.zeros_like(safe_value))
    return safe_value.sum() / mask_b.float().sum().clamp_min(eps)


def _binary_target(x: torch.Tensor) -> torch.Tensor:
    """Finite [0,1] target for CUDA BCE kernels."""
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _nonnegative_weight(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _safe_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _zero_like_loss(ref: torch.Tensor | dict | None = None, *, device=None) -> torch.Tensor:
    """Finite scalar zero for optional/disabled loss branches.

    Avoid expressions like ``tensor.sum() * 0``: if ``tensor`` contains NaN,
    the placeholder becomes NaN and poisons the total loss even when the
    corresponding loss weight is zero.
    """
    if isinstance(ref, torch.Tensor):
        return torch.zeros((), dtype=torch.float32, device=ref.device)
    if isinstance(ref, dict):
        for v in ref.values():
            if torch.is_tensor(v):
                return torch.zeros((), dtype=torch.float32, device=v.device)
    if device is not None:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.zeros((), dtype=torch.float32)


def _masked_neg_score_logits(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """AMP-safe logits for selecting low-cost planner candidates.

    Model outputs can be fp16 under autocast.  Constants such as -1e9 cannot be
    represented in fp16 and may crash before type promotion.  Convert scores to
    fp32 first, then apply a finite fp32 mask value.
    """
    scores_f = _safe_float(scores)
    mask_b = mask.bool()
    fill = torch.full_like(scores_f, -1.0e9)
    return torch.where(mask_b, -scores_f, fill)


def _masked_softmax_low_score(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over valid candidates only, with all-invalid rows set to zero."""
    mask_b = mask.bool()
    logits = _masked_neg_score_logits(scores, mask_b)
    prob = F.softmax(logits, dim=-1)
    prob = torch.where(mask_b, prob, torch.zeros_like(prob))
    denom = prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    row_valid = mask_b.any(dim=-1, keepdim=True)
    prob = torch.where(row_valid, prob / denom, torch.zeros_like(prob))
    return prob


def _pairwise_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """Pairwise ADE [B,A,M_pred,M_gt], computed once and reused."""
    pred_f = _safe_float(pred_traj)
    gt_f = _safe_float(gt_traj)
    return torch.linalg.norm(pred_f[:, :, :, None, :, :2] - gt_f[:, :, None, :, :, :2], dim=-1).mean(dim=-1)


def _focal_bce_values(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    target = _binary_target(target)
    logits = _safe_float(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)
    w = torch.where(target > 0.5, alpha, 1.0 - alpha) * (1.0 - pt).pow(gamma)
    return bce * w


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    return masked_mean(_focal_bce_values(logits, target, gamma=gamma, alpha=alpha), mask)


def pair_mined_focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    max_pos_per_scene: int = 16,
    max_neg_per_scene: int = 48,
    neg_pos_ratio: int = 3,
    min_neg_per_scene: int = 8,
) -> torch.Tensor:
    """Focal BCE with pair-level positive / hard-negative mining."""
    values = _focal_bce_values(logits, target, gamma=gamma, alpha=alpha)
    keep = torch.zeros_like(mask, dtype=torch.bool)
    B = values.shape[0]
    flat_values = values.reshape(B, -1)
    flat_mask = mask.reshape(B, -1).bool()
    flat_target = _binary_target(target).reshape(B, -1) > 0.5
    flat_keep = keep.reshape(B, -1)
    for b in range(B):
        pos = torch.where(flat_mask[b] & flat_target[b])[0]
        neg = torch.where(flat_mask[b] & ~flat_target[b])[0]
        if pos.numel() > 0:
            if max_pos_per_scene > 0 and pos.numel() > max_pos_per_scene:
                _, top = torch.topk(flat_values[b, pos], k=max_pos_per_scene)
                pos = pos[top]
            flat_keep[b, pos] = True
        neg_budget = int(max(min_neg_per_scene, neg_pos_ratio * max(int(pos.numel()), 1)))
        if max_neg_per_scene > 0:
            neg_budget = min(neg_budget, max_neg_per_scene)
        neg_budget = min(neg_budget, int(neg.numel()))
        if neg_budget > 0:
            _, top = torch.topk(flat_values[b, neg], k=neg_budget)
            flat_keep[b, neg[top]] = True
    if keep.any():
        return values[keep].mean()
    return masked_mean(values, mask)



def balanced_bce_all_pairs(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_pos_weight: float = 8.0,
) -> torch.Tensor:
    """Class-balanced BCE over the complete valid pair distribution.

    Pair mining is useful for localization, but by itself changes the effective
    witness prior and can admit a nearly constant high-probability solution.
    This full-distribution term restores calibration while retaining the mined
    focal term for hard examples.
    """
    mask_b = mask.bool()
    if not mask_b.any():
        return _zero_like_loss(logits)
    y = _binary_target(target)
    pos = ((y > 0.5) & mask_b).float().sum()
    neg = ((y <= 0.5) & mask_b).float().sum()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(max_pos_weight))
    values = F.binary_cross_entropy_with_logits(_safe_float(logits), y, reduction="none")
    values = torch.where(y > 0.5, values * pos_weight, values)
    return masked_mean(values, mask_b)


def witness_pair_ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 1.0,
    max_pairs_per_class: int = 64,
) -> torch.Tensor:
    """Force positive witness logits above negatives within the same root scene."""
    y = _binary_target(target) > 0.5
    mask_b = mask.bool()
    logits_f = _safe_float(logits)
    terms = []
    for b in range(logits_f.shape[0]):
        pos = logits_f[b][mask_b[b] & y[b]].reshape(-1)
        neg = logits_f[b][mask_b[b] & ~y[b]].reshape(-1)
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if max_pairs_per_class > 0:
            pos = pos[:max_pairs_per_class]
            neg = neg[:max_pairs_per_class]
        terms.append(torch.relu(float(margin) - pos[:, None] + neg[None, :]).mean())
    return torch.stack(terms).mean() if terms else _zero_like_loss(logits)


def witness_candidate_consistency_loss(
    logits: torch.Tensor,
    opr: torch.Tensor,
    batch: dict[str, torch.Tensor],
    pair_mask: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Link pair witnesses and option preservation to candidate-level labels."""
    cand_mask = batch["cowp/candidates/valid"].bool()
    crit_mask = batch["cowp/critical/valid"].bool()
    pair_prob = torch.sigmoid(_safe_float(logits))
    pair_prob = torch.where(pair_mask, pair_prob.clamp(1e-5, 1.0 - 1e-5), torch.zeros_like(pair_prob))
    # Smooth noisy-OR is less brittle than max and gives every critical pair a gradient.
    candidate_prob = 1.0 - torch.prod(1.0 - pair_prob, dim=-1)
    candidate_prob = candidate_prob.clamp(1e-5, 1.0 - 1e-5)
    candidate_logit = torch.logit(candidate_prob)
    fs_target = _binary_target(batch.get("cowp/candidates/false_safe", torch.zeros_like(candidate_prob)))
    fs_loss = masked_mean(F.binary_cross_entropy_with_logits(candidate_logit, fs_target, reduction="none"), cand_mask)

    opr_safe = torch.where(crit_mask[:, None, :], _safe_float(opr).clamp(0.0, 1.0), torch.ones_like(opr))
    min_opr = opr_safe.min(dim=-1).values
    alpha = float(weights.get("priority_claim_opr_alpha", weights.get("alpha_opr", 0.35)))
    tau = max(float(weights.get("witness_opr_consistency_tau", 0.08)), 1e-3)
    ncf_logit = (min_opr - alpha) / tau
    ncf_target = _binary_target(batch.get("cowp/candidates/noncoercive_feasible", torch.zeros_like(min_opr)))
    ncf_mask = cand_mask & batch.get("cowp/candidates/conventional_safe", cand_mask).bool()
    ncf_loss = masked_mean(F.binary_cross_entropy_with_logits(ncf_logit, ncf_target, reduction="none"), ncf_mask)
    return fs_loss, ncf_loss


def evidential_binary_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    kl_weight: float = 0.01,
) -> torch.Tensor:
    """Expected beta-Bernoulli NLL plus evidence regularization to a uniform prior."""
    mask_b = mask.bool()
    if not mask_b.any():
        return _zero_like_loss(alpha)
    a = _safe_float(alpha).clamp_min(1.0 + 1e-4)
    b = _safe_float(beta).clamp_min(1.0 + 1e-4)
    y = _binary_target(target)
    strength = a + b
    nll = y * (torch.digamma(strength) - torch.digamma(a)) + (1.0 - y) * (torch.digamma(strength) - torch.digamma(b))
    # Remove evidence for the correct class before KL, as in evidential classification.
    a_tilde = y + (1.0 - y) * a
    b_tilde = (1.0 - y) + y * b
    st = a_tilde + b_tilde
    kl = (
        torch.lgamma(st) - torch.lgamma(a_tilde) - torch.lgamma(b_tilde)
        + (a_tilde - 1.0) * (torch.digamma(a_tilde) - torch.digamma(st))
        + (b_tilde - 1.0) * (torch.digamma(b_tilde) - torch.digamma(st))
    )
    return masked_mean(nll + float(kl_weight) * kl, mask_b)



def witness_scene_prior_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_pairs_per_scene: int = 4,
) -> torch.Tensor:
    """Scene-level calibration for witness existence.

    The pairwise witness head is used as a certificate at inference time.  In the
    current experiments it can collapse to a nearly constant high probability,
    which gives acceptable recall but destroys ranking and turns COWP into either
    conventional-safety selection or universal fallback.  This term constrains
    the average predicted witness rate in each scene to match the pseudo-label
    witness rate over valid candidate-agent pairs.  It is intentionally a
    calibration constraint, not an extra heuristic planner rule.
    """
    mask_b = mask.bool()
    if not mask_b.any():
        return _zero_like_loss(logits)
    prob = torch.sigmoid(_safe_float(logits))
    y = _binary_target(target)
    counts = mask_b.float().sum(dim=(1, 2))
    scene_ok = counts >= int(min_pairs_per_scene)
    if not scene_ok.any():
        return _zero_like_loss(logits)
    pred_rate = (prob * mask_b.float()).sum(dim=(1, 2)) / counts.clamp_min(1.0)
    true_rate = (y * mask_b.float()).sum(dim=(1, 2)) / counts.clamp_min(1.0)
    return F.smooth_l1_loss(pred_rate[scene_ok], true_rate[scene_ok], reduction="mean")


def witness_logit_l2_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Tiny anti-saturation regularizer for certificate logits."""
    mask_b = mask.bool()
    if not mask_b.any():
        return _zero_like_loss(logits)
    return masked_mean(_safe_float(logits).pow(2), mask_b)

def _gt_to_pred_natural_assignment(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Map each unordered GT natural mode to its nearest predicted mode.

    Natural alternatives are trained as a set; raw mode indices have no semantic
    correspondence.  Explicit transport labels must therefore be aligned before
    per-mode BCE/root CE supervision.  The natural decoder is frozen during v9
    transport/planner stages, so a hard nearest-ADE assignment is stable and does
    not need gradients.
    """
    pred_traj = pred.get("_natural_pred_traj")
    gt_traj = batch.get("cowp/natural/traj")
    if pred_traj is None or gt_traj is None:
        return None
    with torch.no_grad():
        d = _pairwise_ade(pred_traj, gt_traj)  # [B,A,M_pred,M_gt]
        assignment = d.argmin(dim=2)
        gt_valid = batch.get("cowp/natural/valid")
        if gt_valid is not None:
            assignment = torch.where(gt_valid.bool(), assignment, torch.zeros_like(assignment))
    return assignment.long()


def _align_pred_modes_to_gt(values: torch.Tensor, assignment: torch.Tensor | None) -> torch.Tensor:
    if assignment is None:
        return values
    # values [B,K,A,M_pred], assignment [B,A,M_gt]
    idx = assignment[:, None, :, :].expand(values.shape[0], values.shape[1], assignment.shape[1], assignment.shape[2])
    return torch.gather(values, dim=-1, index=idx)


def _root_low_safe_target(
    batch: dict[str, torch.Tensor],
    mode_count: int,
) -> torch.Tensor | None:
    """Build a per-natural-root low-burden safe-response existence target.

    The cache stores unordered ego-conditioned response slots and the natural
    root assigned to each slot.  The paper's transport predicate is existential:
    for each natural option, does at least one valid, safe, low-burden response
    remain?  Scatter-reducing the explicit slot labels gives that target without
    loading dense response trajectories or relying on a learned root classifier.
    """
    required = (
        "cowp/response/valid",
        "cowp/response/is_safe",
        "cowp/response/is_low_burden",
        "cowp/transport/response_root_index",
    )
    if any(k not in batch for k in required):
        return None
    valid = batch["cowp/response/valid"].bool()
    low_safe = (
        valid
        & batch["cowp/response/is_safe"].bool()
        & batch["cowp/response/is_low_burden"].bool()
    )
    root = batch["cowp/transport/response_root_index"].long()
    root_in_range = (root >= 0) & (root < int(mode_count))
    safe_root = root.clamp(0, max(int(mode_count) - 1, 0))
    target = torch.zeros(
        *root.shape[:-1], int(mode_count),
        device=root.device,
        dtype=torch.float32,
    )
    # scatter_add is portable across the supported torch versions; thresholding
    # implements an OR/existential reduction and is insensitive to duplicates.
    target.scatter_add_(-1, safe_root, (low_safe & root_in_range).float())
    return (target > 0.0).float()


def _zero_like_pred(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    for v in pred.values():
        if torch.is_tensor(v):
            return _zero_like_loss(v)
    raise ValueError("prediction dict contains no tensors")


def _trajectory_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(_safe_float(pred_traj)[..., :2] - _safe_float(gt_traj)[..., :2], dim=-1).mean(dim=-1)


def _weighted_set_minade_from_pairwise(pairwise_ade: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Unordered set supervision for natural alternatives."""
    if not valid.any():
        return _zero_like_loss(ref)
    d_min = pairwise_ade.min(dim=2).values
    w = _nonnegative_weight(weight) * valid.float()
    if w.sum() <= 0:
        return _zero_like_loss(ref)
    return torch.where(valid.bool(), d_min * w, torch.zeros_like(d_min)).sum() / w.sum().clamp_min(1e-6)


def _natural_mixture_nll_from_pairwise(pairwise_ade: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor, ref: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """Order-invariant mixture supervision for natural alternatives."""
    if not valid.any():
        return _zero_like_loss(ref)
    logp = F.log_softmax(_safe_float(logits), dim=-1)[:, :, :, None]
    log_cover = torch.logsumexp(logp - pairwise_ade / max(float(tau), 1e-6), dim=2)
    w = _nonnegative_weight(weight) * valid.float()
    if w.sum() <= 0:
        return _zero_like_loss(ref)
    return -(log_cover * w).sum() / w.sum().clamp_min(1e-6)


def _natural_source_distribution_loss(
    logits: torch.Tensor,
    source_logits: torch.Tensor,
    gt_source: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    source_count: int,
) -> torch.Tensor:
    if not valid.any():
        return _zero_like_loss(logits)
    logits_f = _safe_float(logits)
    source_logits_f = _safe_float(source_logits)
    mix = F.softmax(logits_f, dim=-1)
    src_prob = F.softmax(source_logits_f, dim=-1)
    pred_src = (mix.unsqueeze(-1) * src_prob).sum(dim=2).clamp_min(1e-8)  # [B,A,S]
    pred_log = torch.log(pred_src)

    target = torch.zeros(*gt_source.shape[:2], source_count, device=logits.device, dtype=pred_src.dtype)
    # Clamp corrupted/padded labels before scatter_add_; CUDA scatter kernels
    # assert on out-of-range indices.
    src = torch.nan_to_num(gt_source.float(), nan=float(NaturalSource.PAD), posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD)).long()
    src = src.clamp(0, source_count - 1)
    src_weight = _nonnegative_weight(weight).to(dtype=target.dtype) * valid.to(dtype=target.dtype)
    target.scatter_add_(-1, src, src_weight)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    agent_mask = valid.any(dim=-1) & (src_weight.sum(dim=-1) > 0)
    ce = -(target * pred_log).sum(dim=-1)
    return masked_mean(ce, agent_mask)


def _natural_priority_expectation_loss(logits: torch.Tensor, priority_logits: torch.Tensor, gt_priority: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Priority-preservation supervision for the natural mixture.

    CUDA AMP forbids probability-space ``binary_cross_entropy``/``BCELoss``.
    The decoder predicts mode logits and per-mode priority logits; the paper's
    branch-level priority objective is the expected priority probability over
    mixture modes.  We compute that expectation, convert it back to a finite logit,
    and use ``binary_cross_entropy_with_logits`` so the loss is AMP-safe.
    """
    if not valid.any():
        return _zero_like_loss(logits)
    logits_f = _safe_float(logits)
    priority_logits_f = _safe_float(priority_logits)
    mix = F.softmax(logits_f, dim=-1)
    pred_p = (mix * torch.sigmoid(priority_logits_f)).sum(dim=-1).clamp(1e-6, 1.0 - 1e-6)
    pred_logit = torch.logit(pred_p)

    w = _nonnegative_weight(weight) * valid.float()
    prio = _binary_target(gt_priority)
    tgt = (prio * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1e-6)
    tgt = tgt.clamp(0.0, 1.0)
    agent_mask = valid.any(dim=-1) & (w.sum(dim=-1) > 0)
    bce = F.binary_cross_entropy_with_logits(pred_logit, tgt, reduction="none")
    return masked_mean(bce, agent_mask)


def _branch_minade_from_pairwise(pairwise_ade: torch.Tensor, valid: torch.Tensor, source: torch.Tensor, branch: int, ref: torch.Tensor) -> torch.Tensor:
    branch_gt = valid & (source == int(branch))
    if not branch_gt.any():
        return _zero_like_loss(ref)
    d_min = pairwise_ade.min(dim=2).values
    return masked_mean(d_min, branch_gt)


def _diversity_loss(pred_traj: torch.Tensor, crit_mask: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    B, A, M = pred_traj.shape[:3]
    if M <= 1 or not crit_mask.any():
        return _zero_like_loss(pred_traj)
    xy = _safe_float(pred_traj)[..., :2]
    d = torch.linalg.norm(xy[:, :, :, None, :, :] - xy[:, :, None, :, :, :], dim=-1).mean(dim=-1)
    eye = torch.eye(M, device=pred_traj.device, dtype=torch.bool)[None, None]
    pair_mask = crit_mask[:, :, None, None] & ~eye
    collapse = torch.exp(-d / max(float(tau), 1e-6))
    return masked_mean(collapse, pair_mask)


def natural_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Supervise natural alternatives with branch-aware losses."""
    valid = batch["cowp/natural/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    mask = valid & crit[:, :, None]
    if mask.any():
        gt_traj = _safe_float(batch["cowp/natural/traj"])
        gt_source = torch.nan_to_num(batch["cowp/natural/source"].float(), nan=float(NaturalSource.PAD), posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD)).long()
        gt_source = gt_source.clamp(0, int(NaturalSource.PAD))
        nat_weight = _nonnegative_weight(batch["cowp/natural/weight"])
        pairwise_ade = _pairwise_ade(pred["traj"], gt_traj)

        traj = _weighted_set_minade_from_pairwise(pairwise_ade, mask, nat_weight, pred["traj"])
        logits = pred["logits"]
        mode = _natural_mixture_nll_from_pairwise(
            pairwise_ade,
            logits,
            mask,
            nat_weight,
            pred["traj"],
            tau=float(weights.get("natural_mode_tau_m", 2.0)),
        )

        if "source_logits" in pred:
            source_ce = _natural_source_distribution_loss(
                logits,
                pred["source_logits"],
                gt_source,
                mask,
                nat_weight,
                int(pred["source_logits"].shape[-1]),
            )
        else:
            source_ce = _zero_like_loss(pred["traj"])
        if "priority_logits" in pred:
            priority_loss = _natural_priority_expectation_loss(
                logits,
                pred["priority_logits"],
                batch["cowp/natural/priority_preserved"],
                mask,
                nat_weight,
            )
        else:
            priority_loss = _zero_like_loss(pred["traj"])

        obs_minade = _branch_minade_from_pairwise(pairwise_ade, mask, gt_source, int(NaturalSource.OBS), pred["traj"])
        neu_minade = _branch_minade_from_pairwise(pairwise_ade, mask, gt_source, int(NaturalSource.NEU), pred["traj"])
        prio_minade = _branch_minade_from_pairwise(pairwise_ade, mask, gt_source, int(NaturalSource.PRIO), pred["traj"])
        w_obs = float(weights.get("obs_prediction", 1.0))
        w_neu = float(weights.get("neutral", 0.5))
        w_prio = float(weights.get("priority_rule", 0.5))
        w_sum = max(w_obs + w_neu + w_prio, 1e-6)
        branch_minade = (w_obs * obs_minade + w_neu * neu_minade + w_prio * prio_minade) / w_sum

        neu_gt_mask = mask & (gt_source == int(NaturalSource.NEU))
        if neu_gt_mask.any() and "source_logits" in pred:
            prob_neu = F.softmax(_safe_float(pred["source_logits"]), dim=-1)[..., int(NaturalSource.NEU)] * crit[:, :, None].float()
            pred_mu = (_safe_float(pred["traj"]) * prob_neu[..., None, None]).sum(dim=2) / prob_neu.sum(dim=2).clamp_min(1e-6)[..., None, None]
            gt_w = nat_weight * neu_gt_mask.float()
            gt_mu = (gt_traj * gt_w[..., None, None]).sum(dim=2) / gt_w.sum(dim=2).clamp_min(1e-6)[..., None, None]
            agent_has_neu = neu_gt_mask.any(dim=-1)
            neu_cons = masked_mean(_trajectory_ade(pred_mu, gt_mu), agent_has_neu)
        else:
            neu_cons = _zero_like_loss(pred["traj"])
        div = _diversity_loss(pred["traj"], crit & mask.any(dim=-1), tau=float(weights.get("natural_diversity_tau", 4.0)))
    else:
        traj = _zero_like_loss(pred["traj"])
        mode = _zero_like_loss(pred.get("logits", pred["traj"]))
        source_ce = priority_loss = branch_minade = neu_cons = div = _zero_like_loss(pred["traj"])
        obs_minade = neu_minade = prio_minade = _zero_like_loss(pred["traj"])

    total = (
        weights.get("natural_traj_l1", weights.get("obs_prediction", 1.0)) * traj
        + weights.get("natural_mode_ce", 0.5) * mode
        + weights.get("branch_source_ce", 0.4) * source_ce
        + weights.get("branch_minade", 0.7) * branch_minade
        + weights.get("priority_preservation", 0.4) * priority_loss
        + weights.get("neutral_consistency", 0.3) * neu_cons
        + weights.get("diversity_loss", weights.get("diversity", 0.05)) * div
    )
    return {
        "loss": total,
        "traj": traj,
        "mode": mode,
        "source": source_ce,
        "priority": priority_loss,
        "branch_minade": branch_minade,
        "obs_minade": obs_minade,
        "neutral_minade": neu_minade,
        "prio_minade": prio_minade,
        "neutral_consistency": neu_cons,
        "diversity": div,
    }


def witness_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    cand_mask = batch["cowp/candidates/valid"].bool()
    crit_mask = batch["cowp/critical/valid"].bool()
    pair_mask = cand_mask[:, :, None] & crit_mask[:, None, :]
    y = _binary_target(batch["cowp/witness/exists"])
    mined = pair_mined_focal_bce_with_logits(
        pred["exist_logits"],
        y,
        pair_mask,
        gamma=float(weights.get("witness_focal_gamma", 2.0)),
        alpha=float(weights.get("witness_focal_alpha", 0.25)),
        max_pos_per_scene=int(weights.get("witness_mining_max_pos_per_scene", 16)),
        max_neg_per_scene=int(weights.get("witness_mining_max_neg_per_scene", 48)),
        neg_pos_ratio=int(weights.get("witness_mining_neg_pos_ratio", 3)),
        min_neg_per_scene=int(weights.get("witness_mining_min_neg_per_scene", 8)),
    )
    balanced = balanced_bce_all_pairs(
        pred["exist_logits"], y, pair_mask,
        max_pos_weight=float(weights.get("witness_max_pos_weight", 8.0)),
    )
    pair_rank = witness_pair_ranking_loss(
        pred["exist_logits"], y, pair_mask,
        margin=float(weights.get("witness_pair_margin", 1.0)),
        max_pairs_per_class=int(weights.get("witness_rank_max_per_class", 64)),
    )
    exist = (
        float(weights.get("witness_mined_fraction", 0.35)) * mined
        + float(weights.get("witness_balanced_fraction", 0.65)) * balanced
    )
    pos_mask = pair_mask & (y > 0.5)
    if pos_mask.any():
        token_target = torch.nan_to_num(
            batch["cowp/witness/token"].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).long().clamp(0, pred["token_logits"].shape[-1] - 1)
        token = F.cross_entropy(_safe_float(pred["token_logits"])[pos_mask], token_target[pos_mask], reduction="mean")
        burden = F.smooth_l1_loss(_safe_float(pred["burden_total"])[pos_mask], _safe_float(batch["cowp/witness/burden_total"])[pos_mask], reduction="mean")
        interval = F.smooth_l1_loss(_safe_float(pred["conflict_interval"])[pos_mask], _safe_float(batch["cowp/witness/conflict_interval"])[pos_mask], reduction="mean")
        if "cowp/witness/burden_components" in batch and "burden_components" in pred:
            comps = F.smooth_l1_loss(_safe_float(pred["burden_components"])[pos_mask], _safe_float(batch["cowp/witness/burden_components"])[pos_mask], reduction="mean")
        else:
            comps = _zero_like_loss(pred["exist_logits"])
    else:
        token = _zero_like_loss(pred["exist_logits"])
        burden = _zero_like_loss(pred["exist_logits"])
        interval = _zero_like_loss(pred["exist_logits"])
        comps = _zero_like_loss(pred["exist_logits"])
    opr = masked_mean(torch.abs(_safe_float(pred["opr"]) - _binary_target(batch["cowp/witness/opr"])), pair_mask)
    ci = masked_mean(torch.abs(_safe_float(pred["c_i"]) - _safe_float(batch["cowp/witness/c_i"])), pair_mask)
    candidate_fs, candidate_ncf = witness_candidate_consistency_loss(pred["exist_logits"], pred["opr"], batch, pair_mask, weights)
    if "evidence_alpha" in pred and "evidence_beta" in pred:
        evidence = evidential_binary_loss(
            pred["evidence_alpha"], pred["evidence_beta"], y, pair_mask,
            kl_weight=float(weights.get("witness_evidence_kl", 0.01)),
        )
    else:
        evidence = _zero_like_loss(pred["exist_logits"])
    scene_prior = witness_scene_prior_loss(
        pred["exist_logits"], y, pair_mask,
        min_pairs_per_scene=int(weights.get("witness_scene_prior_min_pairs", 4)),
    )
    logit_l2 = witness_logit_l2_loss(pred["exist_logits"], pair_mask)
    total = (
        weights.get("witness_exist", 2.0) * exist
        + weights.get("witness_pair_rank", 0.5) * pair_rank
        + weights.get("witness_candidate_false_safe", 0.75) * candidate_fs
        + weights.get("witness_candidate_ncf", 0.5) * candidate_ncf
        + weights.get("witness_evidential", 0.25) * evidence
        + weights.get("witness_scene_prior", 0.0) * scene_prior
        + weights.get("witness_logit_l2", 0.0) * logit_l2
        + weights.get("witness_token", 1.0) * token
        + weights.get("witness_burden", 0.5) * burden
        + weights.get("witness_burden_components", 0.25) * comps
        + weights.get("witness_conflict", 0.3) * interval
        + weights.get("witness_opr", 0.5) * (opr + ci)
    )
    return {
        "loss": total, "exist": exist, "exist_mined": mined, "exist_balanced": balanced,
        "pair_rank": pair_rank, "candidate_fs": candidate_fs, "candidate_ncf": candidate_ncf,
        "evidence": evidence, "scene_prior": scene_prior, "logit_l2": logit_l2,
        "token": token, "burden": burden, "components": comps,
        "interval": interval, "opr": opr, "ci": ci,
    }


def response_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    mask = batch["cowp/response/valid"].bool()
    if "cowp/critical/valid" in batch:
        mask = mask & batch["cowp/critical/valid"].bool()[:, None, :, None]
    safe = F.binary_cross_entropy_with_logits(_safe_float(pred["safe_logits"]), _binary_target(batch["cowp/response/is_safe"]), reduction="none")
    low = F.binary_cross_entropy_with_logits(_safe_float(pred["low_logits"]), _binary_target(batch["cowp/response/is_low_burden"]), reduction="none")
    b = torch.abs(_safe_float(pred["burden_total"]) - _safe_float(batch["cowp/response/burden_total"]))
    loss_safe = masked_mean(safe, mask)
    loss_low = masked_mean(low, mask)
    loss_b = masked_mean(b, mask)
    valid_logits = pred.get("valid_logits")
    if valid_logits is not None:
        valid_loss = F.binary_cross_entropy_with_logits(_safe_float(valid_logits), mask.float(), reduction="none")
        # valid supervision includes padded response slots for otherwise valid pairs.
        pair_mask = batch["cowp/candidates/valid"].bool()[:, :, None, None] & batch["cowp/critical/valid"].bool()[:, None, :, None]
        loss_valid = masked_mean(valid_loss, pair_mask)
    else:
        loss_valid = _zero_like_loss(pred["safe_logits"])
    if "source_logits" in pred and "cowp/response/source" in batch and mask.any():
        source_target = torch.nan_to_num(batch["cowp/response/source"].float(), nan=0.0).long().clamp(0, pred["source_logits"].shape[-1] - 1)
        loss_source = F.cross_entropy(_safe_float(pred["source_logits"])[mask], source_target[mask], reduction="mean")
    else:
        loss_source = _zero_like_loss(pred["safe_logits"])

    # Same-root response transport labels are produced by
    # 26_augment_transport_labels.  Root assignment is supervised on every
    # valid response and the minimum-burden marker focuses the bank on the
    # response that actually certifies feasibility for a root option.
    if "root_logits" in pred and "cowp/transport/response_root_index" in batch and mask.any():
        root_target_gt = batch["cowp/transport/response_root_index"].long()
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        if assignment is not None:
            gather_map = assignment[:, None, :, :].expand(
                root_target_gt.shape[0], root_target_gt.shape[1], assignment.shape[1], assignment.shape[2]
            )
            safe_gt = root_target_gt.clamp(0, assignment.shape[-1] - 1)
            # gather_map: [B, K, A, M_gt], safe_gt: [B, K, A, R].
            # torch.gather requires input and index to have the same rank; the
            # previous unsqueeze created a 5-D index and crashed on the first
            # training batch.  Gathering directly maps every GT root id to the
            # aligned predicted-natural-mode id and returns [B, K, A, R].
            root_target = torch.gather(gather_map, -1, safe_gt)
        else:
            root_target = root_target_gt
        root_target = root_target.clamp(0, pred["root_logits"].shape[-1] - 1)
        loss_root = F.cross_entropy(_safe_float(pred["root_logits"])[mask], root_target[mask], reduction="mean")
    else:
        loss_root = _zero_like_loss(pred["safe_logits"])
    if "min_burden_logits" in pred and "cowp/transport/response_is_min_burden" in batch:
        min_target = _binary_target(batch["cowp/transport/response_is_min_burden"])
        min_loss = F.binary_cross_entropy_with_logits(_safe_float(pred["min_burden_logits"]), min_target, reduction="none")
        loss_min = masked_mean(min_loss, mask)
    else:
        loss_min = _zero_like_loss(pred["safe_logits"])

    # ``cowp/response/traj`` is the largest Stage-B target.  The original code
    # computed an L1 tensor for every padded candidate/agent/response slot and
    # masked it afterwards.  That is mathematically equivalent but can allocate
    # hundreds of MB per batch.  Indexing valid slots first preserves the exact
    # supervised objective while avoiding work on padded labels.
    traj_w = float(weights.get("response_traj_l1", 0.1))
    if traj_w != 0.0 and "cowp/response/traj" in batch and mask.any():
        pred_traj = _safe_float(pred["traj"])[mask]
        gt_traj = _safe_float(batch["cowp/response/traj"])[mask]
        loss_traj = torch.abs(pred_traj - gt_traj).mean(dim=(-1, -2)).mean()
    else:
        loss_traj = _zero_like_loss(pred["safe_logits"])

    comps_w = float(weights.get("response_components_l1", 0.25))
    if comps_w != 0.0 and "cowp/response/burden_components" in batch and "burden_components" in pred and mask.any():
        pred_comps = _safe_float(pred["burden_components"])[mask]
        gt_comps = _safe_float(batch["cowp/response/burden_components"])[mask]
        loss_comps = torch.abs(pred_comps - gt_comps).mean(dim=-1).mean()
    else:
        loss_comps = _zero_like_loss(pred["safe_logits"])
    total = (
        weights.get("response_safe_bce", 1.0) * loss_safe
        + weights.get("response_low_bce", 1.0) * loss_low
        + weights.get("response_burden_l1", 0.5) * loss_b
        + weights.get("response_valid_bce", 0.5) * loss_valid
        + weights.get("response_source_ce", 0.2) * loss_source
        + weights.get("response_root_ce", 0.5) * loss_root
        + weights.get("response_min_burden_bce", 0.25) * loss_min
        + comps_w * loss_comps
        + traj_w * loss_traj
    )
    return {"loss": total, "safe": loss_safe, "low": loss_low, "valid": loss_valid, "source": loss_source, "root": loss_root, "min_burden": loss_min, "burden": loss_b, "components": loss_comps, "traj": loss_traj}


def candidate_classification_loss(pred_scores: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Candidate-level auxiliary supervision for learned planner quality."""
    mask = batch["cowp/candidates/valid"].bool()
    ncf = _binary_target(batch["cowp/candidates/noncoercive_feasible"])
    false_safe = _binary_target(batch["cowp/candidates/false_safe"])
    # A collision-free but coercive/false-safe candidate is not a positive NCF
    # example for the planner ranking scalar.  Keeping the raw overlap here made
    # the auxiliary NCF/false-safe losses fight over the same score and produced
    # weak separation in v3.
    ncf = torch.where(false_safe > 0.5, torch.zeros_like(ncf), ncf)
    pred_scores = _safe_float(pred_scores)
    # Lower planner score is better.  Only use the scalar planner score as a
    # binary logit on discriminative candidate labels (NCF or false-safe).  On
    # neutral/ambiguous candidates where both labels are zero, the old two-BCE
    # formulation simultaneously pushed the same score high and low, hurting
    # ranking and calibration.
    disc_mask = mask & ((ncf > 0.5) | (false_safe > 0.5))
    if not disc_mask.any():
        z = _zero_like_loss(pred_scores)
        return {"loss": z, "ncf": z, "false_safe": z}
    ncf_loss = F.binary_cross_entropy_with_logits(-pred_scores, ncf, reduction="none")
    fs_loss = F.binary_cross_entropy_with_logits(pred_scores, false_safe, reduction="none")
    loss_ncf = masked_mean(ncf_loss, disc_mask)
    loss_fs = masked_mean(fs_loss, disc_mask)
    total = weights.get("candidate_ncf_cls", 1.0) * loss_ncf + weights.get("candidate_false_safe_cls", 0.5) * loss_fs
    return {"loss": total, "ncf": loss_ncf, "false_safe": loss_fs}


def candidate_certificate_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Train a *coercion* certificate without conflating it with physical safety.

    ``false_safe`` means that an ego candidate is conventionally collision-free but
    shifts the conflict burden to another road user.  Such a candidate can therefore
    be physically collision-free in a Waymax replay.  Treating every replay-safe
    candidate as an NCF positive creates contradictory within-scene ranking pairs:
    the same false-safe candidate becomes both a positive and a negative.  The v5
    objective did exactly that and the certificate collapsed after a few epochs.

    This objective uses two disjoint semantic classes only:
      * positive: label-level non-coercive feasible and not false-safe;
      * negative: label-level false-safe.

    Collision/offroad supervision remains in ``planner_outcome_supervision`` and is
    used only by the physical feasibility shield at inference.  Ambiguous candidates
    are excluded rather than silently treated as negatives for both concepts.
    """
    mask = batch["cowp/candidates/valid"].bool()
    if not mask.any() or "candidate_ncf_logit" not in pred or "candidate_false_safe_logit" not in pred:
        z = _zero_like_loss(batch["cowp/candidates/valid"])
        return {
            "loss": z, "ncf": z, "false_safe": z, "quality": z,
            "prior": z, "rank": z, "risk_bce": z, "risk_rank": z,
            "spread": z, "overlap_rate": z,
        }

    raw_ncf = _binary_target(batch.get(
        "cowp/candidates/noncoercive_feasible",
        torch.zeros_like(pred["candidate_ncf_logit"]),
    )) > 0.5
    raw_fs = _binary_target(batch.get(
        "cowp/candidates/false_safe",
        torch.zeros_like(pred["candidate_false_safe_logit"]),
    )) > 0.5

    # False-safe dominates only to repair noisy legacy caches.  In correctly built
    # labels the two classes are already mutually exclusive.
    overlap = mask & raw_ncf & raw_fs
    ncf_pos = mask & raw_ncf & ~raw_fs
    fs_pos = mask & raw_fs
    disc = ncf_pos | fs_pos

    ncf_logits = _safe_float(pred["candidate_ncf_logit"])
    fs_logits = _safe_float(pred["candidate_false_safe_logit"])
    quality_logits = _safe_float(pred.get("candidate_quality_logit", ncf_logits - fs_logits))

    if not disc.any():
        z = _zero_like_loss(ncf_logits)
        return {
            "loss": z, "ncf": z, "false_safe": z, "quality": z,
            "prior": z, "rank": z, "risk_bce": z, "risk_rank": z,
            "spread": z, "overlap_rate": overlap.float().mean(),
        }

    ncf_target = ncf_pos.float()
    fs_target = fs_pos.float()
    max_pw = float(weights.get("candidate_cert_max_pos_weight", 4.0))

    ncf_pos_n = ncf_pos.float().sum()
    ncf_neg_n = fs_pos.float().sum()
    fs_pos_n = fs_pos.float().sum()
    fs_neg_n = ncf_pos.float().sum()
    ncf_pw = (ncf_neg_n / ncf_pos_n.clamp_min(1.0)).clamp(1.0, max_pw)
    fs_pw = (fs_neg_n / fs_pos_n.clamp_min(1.0)).clamp(1.0, max_pw)

    ncf_loss = masked_mean(
        F.binary_cross_entropy_with_logits(ncf_logits, ncf_target, reduction="none", pos_weight=ncf_pw),
        disc,
    )
    fs_loss = masked_mean(
        F.binary_cross_entropy_with_logits(fs_logits, fs_target, reduction="none", pos_weight=fs_pw),
        disc,
    )
    q_loss = masked_mean(
        F.binary_cross_entropy_with_logits(quality_logits, ncf_target, reduction="none"),
        disc,
    )

    # Scene-level class-mass calibration is computed over discriminative candidates,
    # not all padded/ambiguous candidates.  This avoids an all-low solution caused by
    # the large ambiguous class.
    disc_count = disc.float().sum(dim=1).clamp_min(1.0)
    gt_ncf_rate = ncf_pos.float().sum(dim=1) / disc_count
    gt_fs_rate = fs_pos.float().sum(dim=1) / disc_count
    pred_ncf_rate = (torch.sigmoid(ncf_logits) * disc.float()).sum(dim=1) / disc_count
    pred_fs_rate = (torch.sigmoid(fs_logits) * disc.float()).sum(dim=1) / disc_count
    scene_has_disc = disc.any(dim=1)
    if scene_has_disc.any():
        prior = (
            F.smooth_l1_loss(pred_ncf_rate[scene_has_disc], gt_ncf_rate[scene_has_disc], reduction="mean")
            + F.smooth_l1_loss(pred_fs_rate[scene_has_disc], gt_fs_rate[scene_has_disc], reduction="mean")
        )
    else:
        prior = _zero_like_loss(ncf_logits)

    # A single semantic risk logit is used by the selector.  Positive values mean
    # more likely coercive.  All ranking masks are explicitly disjoint.
    risk_logit = fs_logits - ncf_logits - 0.5 * quality_logits
    margin = float(weights.get("candidate_cert_pair_margin", 0.5))
    rank_terms: list[torch.Tensor] = []
    risk_rank_terms: list[torch.Tensor] = []
    spread_terms: list[torch.Tensor] = []
    max_pairs = int(weights.get("candidate_cert_max_pairs_per_scene", 256))
    for b in range(mask.shape[0]):
        pos = torch.where(ncf_pos[b])[0]
        neg = torch.where(fs_pos[b])[0]
        if not (pos.numel() and neg.numel()):
            continue
        # Bound O(K^2) memory while keeping deterministic class coverage.
        if pos.numel() * neg.numel() > max_pairs:
            p_keep = max(1, int(max_pairs ** 0.5))
            n_keep = max(1, max_pairs // p_keep)
            pos = pos[:p_keep]
            neg = neg[:n_keep]
        rank_terms.append(
            torch.relu(margin - quality_logits[b, pos[:, None]] + quality_logits[b, neg[None, :]]).mean()
        )
        risk_rank_terms.append(
            torch.relu(margin + risk_logit[b, pos[:, None]] - risk_logit[b, neg[None, :]]).mean()
        )
        vals = risk_logit[b, torch.cat([pos, neg])].float()
        if vals.numel() > 2:
            min_spread = float(weights.get("candidate_cert_min_logit_spread", 0.20))
            spread_terms.append(torch.relu(vals.new_tensor(min_spread) - vals.std(unbiased=False)))

    rank = torch.stack(rank_terms).mean() if rank_terms else _zero_like_loss(quality_logits)
    risk_rank = torch.stack(risk_rank_terms).mean() if risk_rank_terms else _zero_like_loss(quality_logits)
    spread = torch.stack(spread_terms).mean() if spread_terms else _zero_like_loss(quality_logits)

    risk_target = fs_pos.float()
    risk_pw = (ncf_pos.float().sum() / fs_pos.float().sum().clamp_min(1.0)).clamp(1.0, max_pw)
    risk_bce = masked_mean(
        F.binary_cross_entropy_with_logits(risk_logit, risk_target, reduction="none", pos_weight=risk_pw),
        disc,
    )

    total = (
        float(weights.get("candidate_certificate_ncf", 1.0)) * ncf_loss
        + float(weights.get("candidate_certificate_false_safe", 1.0)) * fs_loss
        + float(weights.get("candidate_certificate_quality", 1.0)) * q_loss
        + float(weights.get("candidate_certificate_prior", 0.25)) * prior
        + float(weights.get("candidate_certificate_rank", 1.0)) * rank
        + float(weights.get("candidate_certificate_risk_bce", 1.0)) * risk_bce
        + float(weights.get("candidate_certificate_risk_rank", 1.0)) * risk_rank
        + float(weights.get("candidate_certificate_spread", 0.05)) * spread
    )
    return {
        "loss": total,
        "ncf": ncf_loss,
        "false_safe": fs_loss,
        "quality": q_loss,
        "prior": prior,
        "rank": rank,
        "risk_bce": risk_bce,
        "risk_rank": risk_rank,
        "spread": spread,
        "overlap_rate": overlap.float().sum() / mask.float().sum().clamp_min(1.0),
    }

def planner_imitation_loss(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Imitation term using the logged ego candidate when available."""
    scores = _safe_float(scores)
    mask = batch["cowp/candidates/valid"].bool()
    logged = batch.get("cowp/candidates/is_logged")
    if logged is None:
        return _zero_like_loss(scores)
    target_mask = logged.bool() & mask
    losses = []
    for b in range(scores.shape[0]):
        tgt = torch.where(target_mask[b])[0]
        if tgt.numel() == 0:
            continue
        logits = _masked_neg_score_logits(scores[b], mask[b])
        losses.append(F.cross_entropy(logits.unsqueeze(0), tgt[:1]))
    return torch.stack(losses).mean() if losses else _zero_like_loss(scores)




def priority_claim_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """Supervise protected-priority claims for pairwise P-NCF gating.

    A pair is treated as a protected claim when a coercion witness exists and the
    natural response set collapses below the OPR threshold or exceeds the safe
    burden budget.  This is deliberately pair-level, so the planner can hard-veto
    coercing high-priority agents while keeping low-priority negotiations as soft
    costs.
    """
    cand_mask = batch["cowp/candidates/valid"].bool()
    crit_mask = batch["cowp/critical/valid"].bool()
    pair_mask = cand_mask[:, :, None] & crit_mask[:, None, :]
    if logits is None or not pair_mask.any():
        return _zero_like_loss(batch["cowp/candidates/valid"])
    exists = _binary_target(batch.get("cowp/witness/exists", torch.zeros_like(logits))) > 0.5
    opr = _safe_float(batch.get("cowp/witness/opr", torch.ones_like(logits))).clamp(0.0, 1.0)
    alpha = float(weights.get("priority_claim_opr_alpha", weights.get("alpha_opr", 0.35)))
    collapse = opr < alpha
    min_safe = batch.get("cowp/witness/min_safe_burden")
    if min_safe is not None:
        beta_default = float(weights.get("priority_claim_beta", 0.65))
        beta = batch.get("cowp/natural/beta")
        if beta is not None:
            beta_t = _safe_float(beta)
            while beta_t.ndim < min_safe.ndim:
                beta_t = beta_t[:, None, :]
            high_burden = _safe_float(min_safe) > beta_t
        else:
            high_burden = _safe_float(min_safe) > beta_default
    else:
        high_burden = torch.zeros_like(exists)
    target = (exists & (collapse | high_burden)).float()
    values = F.binary_cross_entropy_with_logits(_safe_float(logits), target, reduction="none")
    # Dynamic positive reweighting without exploding when positives are rare.
    pos = (target > 0.5) & pair_mask
    neg = (target <= 0.5) & pair_mask
    if pos.any():
        pos_weight = float(weights.get("priority_claim_pos_weight", 3.0))
        values = torch.where(pos, values * pos_weight, values)
    return masked_mean(values, pair_mask)


def planner_outcome_supervision(outcome: dict[str, torch.Tensor] | None, scores: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Strong closed-loop supervision from attached Waymax candidate outcomes.

    The old expected-cost-only objective is weak when only a few replayed candidates
    are attached. This combines candidate outcome classification, log-divergence
    regression, and safe-vs-unsafe pairwise ranking.
    """
    zero = _zero_like_loss(scores)
    rollout_valid = batch.get("waymax/candidate_rollout_valid")
    collision = batch.get("waymax/candidate_collision")
    offroad = batch.get("waymax/candidate_offroad")
    logdiv = batch.get("waymax/candidate_log_divergence")
    if rollout_valid is None or (collision is None and offroad is None and logdiv is None):
        return {"loss": zero, "cls": zero, "logdiv": zero, "rank": zero, "expected_cost": zero}
    mask = batch["cowp/candidates/valid"].bool() & rollout_valid.bool()
    if not mask.any():
        return {"loss": zero, "cls": zero, "logdiv": zero, "rank": zero, "expected_cost": zero}
    scores_f = _safe_float(scores)
    unsafe = torch.zeros_like(scores_f, dtype=torch.float32)
    if collision is not None:
        unsafe = torch.maximum(unsafe, _binary_target(collision))
    if offroad is not None:
        unsafe = torch.maximum(unsafe, _binary_target(offroad))
    if logdiv is not None:
        ld_target = _safe_float(logdiv).clamp_min(0.0)
        soft_unsafe = (ld_target > float(weights.get("outcome_logdiv_unsafe_threshold", 8.0))).float()
        unsafe = torch.maximum(unsafe, soft_unsafe)
    else:
        ld_target = torch.zeros_like(scores)

    cls = zero
    ldr = zero
    if outcome is not None:
        terms = []
        if collision is not None and "collision_logit" in outcome:
            terms.append(masked_mean(F.binary_cross_entropy_with_logits(_safe_float(outcome["collision_logit"]), _binary_target(collision), reduction="none"), mask))
        if offroad is not None and "offroad_logit" in outcome:
            terms.append(masked_mean(F.binary_cross_entropy_with_logits(_safe_float(outcome["offroad_logit"]), _binary_target(offroad), reduction="none"), mask))
        cls = torch.stack(terms).mean() if terms else zero
        if logdiv is not None and "logdiv" in outcome:
            ldr = masked_mean(F.smooth_l1_loss(_safe_float(outcome["logdiv"]), ld_target.clamp_max(50.0) / 10.0, reduction="none"), mask)

    # Lower planner score should rank safe rollout candidates above unsafe ones.
    rank_terms = []
    for b in range(scores.shape[0]):
        safe_idx = torch.where(mask[b] & (unsafe[b] < 0.5))[0]
        unsafe_idx = torch.where(mask[b] & (unsafe[b] >= 0.5))[0]
        if safe_idx.numel() and unsafe_idx.numel():
            rank_terms.append(torch.relu(float(weights.get("outcome_pair_margin", 1.0)) + scores_f[b, safe_idx[:, None]] - scores_f[b, unsafe_idx[None, :]]).mean())
    rank = torch.stack(rank_terms).mean() if rank_terms else zero

    cost = unsafe + ld_target.clamp_min(0.0) / float(weights.get("outcome_logdiv_scale", 10.0))
    prob = _masked_softmax_low_score(scores_f, mask)
    row_valid = mask.any(dim=-1)
    expected_per_scene = (prob * cost).sum(dim=-1)
    expected = expected_per_scene[row_valid].mean() if row_valid.any() else zero
    total = (
        float(weights.get("outcome_cls", 1.0)) * cls
        + float(weights.get("outcome_logdiv", 0.5)) * ldr
        + float(weights.get("outcome_pair_rank", 1.5)) * rank
        + float(weights.get("outcome_expected_cost", 0.5)) * expected
    )
    return {"loss": total, "cls": cls, "logdiv": ldr, "rank": rank, "expected_cost": expected}


def planner_outcome_loss(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Optional rollout/outcome surrogate when Waymax candidate labels exist."""
    scores = _safe_float(scores)
    mask = batch["cowp/candidates/valid"].bool()
    rollout_valid = batch.get("waymax/candidate_rollout_valid")
    if rollout_valid is not None:
        mask = mask & rollout_valid.bool()
    collision = batch.get("waymax/candidate_collision")
    offroad = batch.get("waymax/candidate_offroad")
    logdiv = batch.get("waymax/candidate_log_divergence")
    if collision is None and offroad is None and logdiv is None:
        return _zero_like_loss(scores)
    if not mask.any():
        return _zero_like_loss(scores)
    cost = torch.zeros_like(scores, dtype=torch.float32)
    if collision is not None:
        cost = cost + _safe_float(collision)
    if offroad is not None:
        cost = cost + _safe_float(offroad)
    if logdiv is not None:
        cost = cost + _safe_float(logdiv).clamp_min(0.0) / 10.0
    prob = _masked_softmax_low_score(scores, mask)
    row_valid = mask.any(dim=-1)
    expected_per_scene = (prob * cost).sum(dim=-1)
    return expected_per_scene[row_valid].mean() if row_valid.any() else _zero_like_loss(scores)


def planner_ranking_loss(scores: torch.Tensor, ncf: torch.Tensor, false_safe: torch.Tensor, cand_mask: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    scores = _safe_float(scores)
    losses = []
    B = scores.shape[0]
    ncf_b = _binary_target(ncf) > 0.5
    false_b = _binary_target(false_safe) > 0.5
    cand_mask = cand_mask.bool()
    for b in range(B):
        pos = torch.where(cand_mask[b] & ncf_b[b])[0]
        neg = torch.where(cand_mask[b] & false_b[b])[0]
        if len(pos) and len(neg):
            # Lower score is better.
            losses.append(torch.relu(margin + scores[b, pos[:, None]] - scores[b, neg[None, :]]).mean())
    return torch.stack(losses).mean() if losses else _zero_like_loss(scores)


def set_transport_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Supervise the explicit set certificate with aggregate set statistics.

    Existing caches already contain OPR, minimum burden and source-wise mass, so
    v8 can be trained without rebuilding WOMD.  A later optional cache augmentation
    can add mode-level labels, but the aggregate objective already prevents the
    certificate from degenerating into a candidate classifier.
    """
    cand = batch["cowp/candidates/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    pair = cand[:, :, None] & crit[:, None, :]
    if not pair.any():
        z = _zero_like_loss(pred)
        return {"loss": z, "witness": z, "opr": z, "burden": z, "conflict": z, "source": z, "response": z}

    y = _binary_target(batch["cowp/witness/exists"])
    witness = focal_bce_with_logits(
        pred["exist_logits"], y, pair,
        gamma=float(weights.get("set_transport_focal_gamma", 2.0)),
        alpha=float(weights.get("set_transport_focal_alpha", 0.5)),
    )
    opr = masked_mean(torch.abs(_safe_float(pred["opr"]) - _binary_target(batch["cowp/witness/opr"])), pair)
    burden_target = _safe_float(batch["cowp/witness/burden_total"]).clamp(0.0, 2.0)
    burden = masked_mean(torch.abs(_safe_float(pred["min_safe_burden"]).clamp(0.0, 2.0) - burden_target), pair)

    conflict_target = batch.get("cowp/witness/natural_conflict_mass")
    if conflict_target is None:
        conflict_target = batch.get("cowp/witness/c_i", torch.zeros_like(pred["natural_conflict_mass"]))
    conflict = masked_mean(torch.abs(_safe_float(pred["natural_conflict_mass"]) - _safe_float(conflict_target).clamp(0.0, 1.0)), pair)

    src_terms = []
    src_mask = pair[..., None]
    for key, pkey in (
        ("cowp/witness/natural_conflict_mass_by_source", "conflict_mass_by_source"),
        ("cowp/witness/low_safe_mass_by_source", "low_safe_mass_by_source"),
    ):
        target = batch.get(key)
        if target is not None and pkey in pred:
            src_terms.append(masked_mean(torch.abs(_safe_float(pred[pkey]) - _safe_float(target).clamp(0.0, 1.0)), src_mask))
    source = torch.stack(src_terms).mean() if src_terms else _zero_like_loss(pred)

    if all(k in batch for k in ("cowp/response/is_safe", "cowp/response/is_low_burden", "cowp/response/valid")):
        low_exists_target = ((batch["cowp/response/is_safe"].bool() & batch["cowp/response/is_low_burden"].bool() & batch["cowp/response/valid"].bool()).any(dim=-1)).float()
        response_prob = _safe_float(pred["response_exist_low_safe"]).clamp(1.0e-5, 1.0 - 1.0e-5)
        response_logit = torch.logit(response_prob)
        response = masked_mean(
            F.binary_cross_entropy_with_logits(response_logit, low_exists_target, reduction="none"),
            pair,
        )
    else:
        response = _zero_like_loss(pred)

    # Explicit per-natural-mode transport supervision removes the aggregate
    # identifiability failure of v8: many arbitrary conflict/retain mode
    # decompositions can reproduce the same OPR and witness scalar.
    mode_valid = batch.get("cowp/transport/mode_valid")
    mode_conflict_target = batch.get("cowp/transport/mode_conflict")
    mode_retain_target = batch.get("cowp/transport/mode_retained_low_safe")
    def _balanced_mode_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = _binary_target(target)
        active = mask.float()
        pos = (target * active).sum()
        neg = ((1.0 - target) * active).sum()
        max_pw = float(weights.get("set_transport_mode_max_pos_weight", 4.0))
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.25, max_pw)
        raw_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
        return masked_mean(raw_loss, mask)

    if mode_valid is not None and mode_conflict_target is not None and "mode_conflict_prob" in pred:
        mm = mode_valid.bool() & pair[..., None]
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        if "mode_conflict_logits" in pred:
            cp_logit = _align_pred_modes_to_gt(_safe_float(pred["mode_conflict_logits"]), assignment)
        else:
            cp = _align_pred_modes_to_gt(_safe_float(pred["mode_conflict_prob"]), assignment).clamp(1.0e-5, 1.0 - 1.0e-5)
            cp_logit = torch.logit(cp)
        mode_conflict = _balanced_mode_bce(cp_logit, mode_conflict_target, mm)
    else:
        mode_conflict = _zero_like_loss(pred)
    if mode_valid is not None and mode_retain_target is not None and "mode_retain_prob" in pred:
        mm = mode_valid.bool() & pair[..., None]
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        if "mode_retain_logits" in pred:
            rp_logit = _align_pred_modes_to_gt(_safe_float(pred["mode_retain_logits"]), assignment)
        else:
            rp = _align_pred_modes_to_gt(_safe_float(pred["mode_retain_prob"]), assignment).clamp(1.0e-5, 1.0 - 1.0e-5)
            rp_logit = torch.logit(rp)
        mode_retain = _balanced_mode_bce(rp_logit, mode_retain_target, mm)
    else:
        mode_retain = _zero_like_loss(pred)
    if (mode_valid is not None and mode_conflict_target is not None and mode_retain_target is not None
            and "mode_uncertainty" in pred):
        mm = mode_valid.bool() & pair[..., None]
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        cp_aligned = _align_pred_modes_to_gt(_safe_float(pred["mode_conflict_prob"]), assignment)
        rp_aligned = _align_pred_modes_to_gt(_safe_float(pred["mode_retain_prob"]), assignment)
        u_aligned = _align_pred_modes_to_gt(_safe_float(pred["mode_uncertainty"]), assignment)
        cp_err = torch.abs(cp_aligned.detach() - _binary_target(mode_conflict_target))
        rp_err = torch.abs(rp_aligned.detach() - _binary_target(mode_retain_target))
        error_target = (0.5 * (cp_err + rp_err)).clamp(0.0, 1.0)
        mode_uncertainty = masked_mean(torch.abs(u_aligned - error_target), mm)
    else:
        mode_uncertainty = _zero_like_loss(pred)

    # Direct primitive-indexed response recovery.  v11 reconstructed this event
    # only through an unordered response bank and a difficult 24-class root CE;
    # the resulting root recovery stayed close to no-skill and became the main
    # feasibility bottleneck.  Supervise the root-indexed transport head from the
    # exact same response-set labels used to construct the dataset.
    root_low_safe_target = _root_low_safe_target(
        batch,
        int(pred["mode_recovery_logits"].shape[-1]) if "mode_recovery_logits" in pred else 0,
    ) if "mode_recovery_logits" in pred else None
    if mode_valid is not None and root_low_safe_target is not None:
        mm = mode_valid.bool() & pair[..., None]
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        recovery_logit = _align_pred_modes_to_gt(
            _safe_float(pred["mode_recovery_logits"]), assignment
        )
        mode_recovery = _balanced_mode_bce(recovery_logit, root_low_safe_target, mm)
    else:
        mode_recovery = _zero_like_loss(pred)

    # Keep the old response-bank/root-classifier path as an auxiliary
    # reconstruction constraint.  It is useful for interpretability and response
    # visualization, but no longer defines the decision certificate.
    if root_low_safe_target is not None and "response_root_exist_aux" in pred and mode_valid is not None:
        mm = mode_valid.bool() & pair[..., None]
        assignment = _gt_to_pred_natural_assignment(pred, batch)
        aux_prob = _align_pred_modes_to_gt(
            _safe_float(pred["response_root_exist_aux"]), assignment
        ).clamp(1.0e-5, 1.0 - 1.0e-5)
        response_root_aux = _balanced_mode_bce(
            torch.logit(aux_prob), root_low_safe_target, mm
        )
    else:
        response_root_aux = _zero_like_loss(pred)

    # Supervise conflict-conditioned same-root recovery mass when augmented
    # labels are available.
    recovery_target = batch.get("cowp/transport/root_recovery_mass")
    if recovery_target is not None and "root_recovery_mass" in pred:
        rt = _safe_float(recovery_target).clamp(0.0, 1.0)
        rp = _safe_float(pred["root_recovery_mass"]).clamp(1.0e-5, 1.0 - 1.0e-5)
        has_recovery = (rt > float(weights.get("set_transport_recovery_positive_floor", 1.0e-4))).float()
        active = pair.float()
        pos = (has_recovery * active).sum()
        neg = ((1.0 - has_recovery) * active).sum()
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(weights.get("set_transport_recovery_max_pos_weight", 8.0)))
        presence = masked_mean(
            F.binary_cross_entropy_with_logits(torch.logit(rp), has_recovery, reduction="none", pos_weight=pos_weight), pair
        )
        pos_mask = pair & (has_recovery > 0.5)
        magnitude = masked_mean(torch.abs(rp - rt), pos_mask) if pos_mask.any() else _zero_like_loss(rp)
        root_recovery = 0.7 * presence + 0.3 * magnitude
    else:
        root_recovery = _zero_like_loss(pred)

    # Candidate-level budget supervision.  Pair AUPRC can be high while an
    # any/max reduction rejects nearly every candidate.  Supervise the monotone
    # BCOT aggregate directly with the same disjoint NCF/false-safe labels used
    # for evaluation.  The aggregate contains no generic candidate latent, so
    # this objective calibrates option transport rather than replacing it.
    candidate_transport = pred.get("candidate_transport_risk")
    candidate_budget_coverage = _zero_like_loss(pred)
    candidate_budget_ncf_rate = _zero_like_loss(pred)
    candidate_budget_false_safe_rate = _zero_like_loss(pred)
    if candidate_transport is not None:
        ncf_target = batch.get("cowp/candidates/noncoercive_feasible")
        fs_target = batch.get("cowp/candidates/false_safe")
        # Missing supervision must not silently become all-negative labels.  The
        # training entry point validates these keys for witness/planner stages;
        # keeping the branch explicit also makes unit tests and custom callers
        # expose the problem instead of reporting a misleading zero loss.
        if ncf_target is None or fs_target is None:
            candidate_budget = _zero_like_loss(candidate_transport)
        else:
            raw_ncf = _binary_target(ncf_target) > 0.5
            raw_fs = _binary_target(fs_target) > 0.5
            ncf_pos = cand & raw_ncf & ~raw_fs
            fs_pos = cand & raw_fs
            disc = ncf_pos | fs_pos
            candidate_budget_coverage = disc.float().sum() / cand.float().sum().clamp_min(1.0)
            candidate_budget_ncf_rate = ncf_pos.float().sum() / disc.float().sum().clamp_min(1.0)
            candidate_budget_false_safe_rate = fs_pos.float().sum() / disc.float().sum().clamp_min(1.0)
            if disc.any():
                risk = _safe_float(candidate_transport).clamp(1.0e-5, 1.0 - 1.0e-5)
                risk_logit = torch.logit(risk)
                target = fs_pos.float()
                pos_weight = (
                    ncf_pos.float().sum() / fs_pos.float().sum().clamp_min(1.0)
                ).clamp(1.0, float(weights.get("set_transport_candidate_max_pos_weight", 4.0)))
                budget_bce = masked_mean(
                    F.binary_cross_entropy_with_logits(
                        risk_logit, target, reduction="none", pos_weight=pos_weight
                    ),
                    disc,
                )
                rank_terms: list[torch.Tensor] = []
                margin = float(weights.get("set_transport_candidate_margin", 0.10))
                max_pairs = int(weights.get("set_transport_candidate_max_pairs", 256))
                for b in range(cand.shape[0]):
                    good = torch.where(ncf_pos[b])[0]
                    bad = torch.where(fs_pos[b])[0]
                    if not (good.numel() and bad.numel()):
                        continue
                    if good.numel() * bad.numel() > max_pairs:
                        g_keep = max(1, int(max_pairs ** 0.5))
                        b_keep = max(1, max_pairs // g_keep)
                        good = good[:g_keep]
                        bad = bad[:b_keep]
                    rank_terms.append(
                        torch.relu(
                            margin + risk[b, good[:, None]] - risk[b, bad[None, :]]
                        ).mean()
                    )
                budget_rank = torch.stack(rank_terms).mean() if rank_terms else _zero_like_loss(risk)
                candidate_budget = 0.65 * budget_bce + 0.35 * budget_rank
            else:
                candidate_budget = _zero_like_loss(candidate_transport)
    else:
        candidate_budget = _zero_like_loss(pred)

    total = (
        float(weights.get("set_transport_witness", 2.0)) * witness
        + float(weights.get("set_transport_opr", 1.0)) * opr
        + float(weights.get("set_transport_burden", 0.75)) * burden
        + float(weights.get("set_transport_conflict", 1.0)) * conflict
        + float(weights.get("set_transport_source", 0.75)) * source
        + float(weights.get("set_transport_response_exist", 0.5)) * response
        + float(weights.get("set_transport_mode_conflict", 1.5)) * mode_conflict
        + float(weights.get("set_transport_mode_retain", 1.5)) * mode_retain
        + float(weights.get("set_transport_mode_uncertainty", 0.25)) * mode_uncertainty
        + float(weights.get("set_transport_mode_recovery", 2.0)) * mode_recovery
        + float(weights.get("set_transport_response_root_aux", 0.15)) * response_root_aux
        + float(weights.get("set_transport_root_recovery", 0.75)) * root_recovery
        + float(weights.get("set_transport_candidate_budget", 2.0)) * candidate_budget
    )
    return {
        "loss": total, "witness": witness, "opr": opr, "burden": burden,
        "conflict": conflict, "source": source, "response": response,
        "mode_conflict": mode_conflict, "mode_retain": mode_retain,
        "mode_uncertainty": mode_uncertainty, "mode_recovery": mode_recovery,
        "response_root_aux": response_root_aux, "root_recovery": root_recovery,
        "candidate_budget": candidate_budget,
        "candidate_budget_coverage": candidate_budget_coverage.detach(),
        "candidate_budget_ncf_rate": candidate_budget_ncf_rate.detach(),
        "candidate_budget_false_safe_rate": candidate_budget_false_safe_rate.detach(),
    }
