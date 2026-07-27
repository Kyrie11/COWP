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


def weighted_masked_mean(
    value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Finite weighted mean over a boolean mask."""
    mask_b = torch.broadcast_to(mask.bool(), value.shape) if mask.shape != value.shape else mask.bool()
    weight_f = torch.broadcast_to(_nonnegative_weight(weight), value.shape) if weight.shape != value.shape else _nonnegative_weight(weight)
    use = mask_b & torch.isfinite(value) & torch.isfinite(weight_f) & (weight_f > 0.0)
    safe = torch.where(use, value.float(), torch.zeros_like(value, dtype=torch.float32))
    w = torch.where(use, weight_f.float(), torch.zeros_like(weight_f, dtype=torch.float32))
    return (safe * w).sum() / w.sum().clamp_min(eps)


def _binary_target(x: torch.Tensor) -> torch.Tensor:
    """Finite [0,1] target for CUDA BCE kernels."""
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _nonnegative_weight(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _safe_float(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _primitive_burden_targets(
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Recover paper-aligned primitive burden from v9-compatible caches.

    Older caches folded option loss into burden component 4.  The manuscript
    explicitly keeps option preservation outside B_prim, so training reconstructs
    the primitive target from response labels (preferred) or by zeroing the
    option component and renormalizing the five physical/normative weights.
    """
    comps = batch.get("cowp/witness/burden_components")
    clean_comps = None
    if comps is not None:
        clean_comps = _safe_float(comps).clone()
        if clean_comps.shape[-1] >= 5:
            clean_comps[..., 4] = 0.0

    response_b = batch.get("cowp/response/burden_total")
    response_valid = batch.get("cowp/response/valid")
    response_safe = batch.get("cowp/response/is_safe")
    if response_b is not None and response_valid is not None and response_safe is not None:
        safe = response_valid.bool() & response_safe.bool()
        b = _safe_float(response_b).clamp(0.0, 2.0)
        masked = torch.where(safe, b, torch.full_like(b, float("inf")))
        best_idx = masked.argmin(dim=-1)
        best = masked.amin(dim=-1)
        best = torch.where(safe.any(dim=-1), best, torch.full_like(best, 2.0))
        response_comps = batch.get("cowp/response/burden_components")
        if response_comps is not None:
            rc = _safe_float(response_comps)
            gather_idx = best_idx[..., None, None].expand(*best_idx.shape, 1, rc.shape[-1])
            clean_comps = torch.gather(rc, -2, gather_idx).squeeze(-2)
            if clean_comps.shape[-1] >= 5:
                clean_comps = clean_comps.clone()
                clean_comps[..., 4] = 0.0
        return best, clean_comps

    if clean_comps is None:
        total = batch.get("cowp/witness/burden_total")
        return (_safe_float(total).clamp(0.0, 2.0) if total is not None else None), None

    # Keep the manuscript's physical/normative coefficients unchanged.  The
    # historical option coefficient remains in the normalization denominator,
    # while its component is exactly zero; renormalizing the remaining five
    # terms would silently increase primitive burden by 1/0.85.
    w = torch.tensor(
        [
            float(weights.get("burden_weight_acc", 0.25)),
            float(weights.get("burden_weight_jerk", 0.10)),
            float(weights.get("burden_weight_progress", 0.20)),
            float(weights.get("burden_weight_risk", 0.20)),
            float(weights.get("burden_weight_option", 0.15)),
            float(weights.get("burden_weight_norm", 0.10)),
        ],
        device=clean_comps.device,
        dtype=clean_comps.dtype,
    )
    w = w[: clean_comps.shape[-1]]
    w = w / w.sum().clamp_min(1.0e-6)
    return (clean_comps[..., : w.numel()] * w).sum(dim=-1).clamp(0.0, 2.0), clean_comps


def _weighted_upper_cvar_target(values: torch.Tensor, weights: torch.Tensor, tail_mass: float) -> torch.Tensor:
    q = min(max(float(tail_mass), 1.0e-3), 1.0)
    w = weights.clamp_min(0.0)
    total = w.sum(dim=-1, keepdim=True)
    w = w / total.clamp_min(1.0e-8)
    v_sorted, order = values.sort(dim=-1, descending=True)
    w_sorted = torch.gather(w, -1, order)
    before = w_sorted.cumsum(dim=-1) - w_sorted
    remaining = torch.relu(torch.as_tensor(q, device=w.device, dtype=w.dtype) - before)
    take = torch.minimum(w_sorted, remaining)
    out = (take * v_sorted).sum(dim=-1) / take.sum(dim=-1).clamp_min(1.0e-8)
    return torch.where(total.squeeze(-1) > 1.0e-8, out, torch.zeros_like(out))


def paper_aligned_supervision_batch(
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Rebuild v16.6 certificate targets from v9 transport primitives.

    This is a shallow-copy adapter: large tensors are not duplicated.  It makes
    old augmented caches obey the manuscript's current equations without a full
    WOMD rebuild: floor-smoothed transported OPR, conflict-conditioned same-root
    CVaR burden, gamma-separated witness existence, and primitive burden with no
    circular option component.
    """
    if float(weights.get("paper_aligned_witness_targets", 1.0)) <= 0.0:
        return batch
    required = (
        "cowp/candidates/valid", "cowp/critical/valid",
        "cowp/natural/weight", "cowp/natural/valid", "cowp/natural/beta",
        "cowp/transport/mode_valid", "cowp/transport/mode_conflict",
        "cowp/transport/mode_retained_low_safe",
        "cowp/transport/response_root_index",
        "cowp/response/valid", "cowp/response/is_safe",
        "cowp/response/burden_total",
    )
    if not all(k in batch for k in required):
        return batch

    cand = batch["cowp/candidates/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    pair = cand[:, :, None] & crit[:, None, :]
    mode_valid = batch["cowp/transport/mode_valid"].bool() & pair[..., None]
    mode_conflict = batch["cowp/transport/mode_conflict"].bool() & mode_valid
    mode_retain = batch["cowp/transport/mode_retained_low_safe"].bool() & mode_valid
    M = int(mode_valid.shape[-1])

    nat_valid = batch["cowp/natural/valid"].bool()[..., :M]
    p = _safe_float(batch["cowp/natural/weight"])[..., :M] * nat_valid.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    eps_p = min(max(float(weights.get("set_transport_probability_floor", 0.02)), 0.0), 0.25)
    valid_count = nat_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    p = ((1.0 - eps_p) * p + eps_p * nat_valid.float() / valid_count) * nat_valid.float()
    p_pair = p[:, None, :, :].expand_as(mode_valid.float())
    conflict_mass = (p_pair * mode_conflict.float()).sum(dim=-1).clamp(0.0, 1.0)
    opr = (p_pair * mode_retain.float()).sum(dim=-1).clamp(0.0, 1.0)

    response_valid = batch["cowp/response/valid"].bool()
    response_safe = batch["cowp/response/is_safe"].bool()
    safe = response_valid & response_safe & pair[..., None]
    response_b = _safe_float(batch["cowp/response/burden_total"]).clamp(0.0, 2.0)
    root_idx = batch["cowp/transport/response_root_index"].long().clamp(0, max(M - 1, 0))
    root_axis = torch.arange(M, device=root_idx.device).view(*([1] * root_idx.ndim), M)
    same_root = safe[..., None] & (root_idx[..., None] == root_axis)
    root_values = torch.where(
        same_root,
        response_b[..., None],
        torch.full((*response_b.shape, M), float("inf"), device=response_b.device, dtype=response_b.dtype),
    )
    root_min = root_values.amin(dim=-2)
    root_min = torch.where(same_root.any(dim=-2), root_min, torch.full_like(root_min, 2.0))

    beta = _safe_float(batch["cowp/natural/beta"])[:, None, :, None]
    root_excess = torch.relu(root_min - beta)
    tail = _weighted_upper_cvar_target(
        root_excess,
        p_pair * mode_conflict.float(),
        float(weights.get("set_transport_cvar_tail_mass", 0.25)),
    )
    delta = float(weights.get("witness_conflict_mass_floor", 0.10))
    gamma = float(weights.get("witness_burden_gamma", 0.10))
    alpha = float(weights.get("witness_opr_alpha", weights.get("priority_claim_opr_alpha", 0.35)))
    exists = pair & (conflict_mass > delta) & ((tail > gamma) | (opr < alpha))

    primitive_total, primitive_comps = _primitive_burden_targets(batch, weights)
    out = dict(batch)
    old_exists = batch.get("cowp/witness/exists")
    out["cowp/witness/explanation_valid"] = exists & (
        _binary_target(old_exists) > 0.5 if old_exists is not None else exists
    )
    out["cowp/witness/exists"] = exists
    out["cowp/witness/opr"] = opr
    out["cowp/witness/natural_conflict_mass"] = conflict_mass
    out["cowp/witness/tail_burden_excess"] = tail
    out["cowp/witness/c_i"] = tail
    out["cowp/witness/root_min_safe_burden"] = root_min
    if primitive_total is not None:
        out["cowp/witness/burden_total"] = primitive_total
        out["cowp/witness/min_safe_burden"] = primitive_total
    if primitive_comps is not None:
        out["cowp/witness/burden_components"] = primitive_comps

    conventional = batch.get("cowp/candidates/conventional_safe")
    if conventional is not None:
        conventional_b = conventional.bool() & cand
        false_safe = conventional_b & (exists & crit[:, None, :]).any(dim=-1)
        pair_good = (tail <= gamma) & (opr >= alpha)
        pair_good = pair_good | ~crit[:, None, :]
        ncf = conventional_b & pair_good.all(dim=-1)
        out["cowp/candidates/false_safe"] = false_safe
        out["cowp/candidates/noncoercive_feasible"] = ncf
    return out


def _safe_velocity_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Avoid the undefined backward derivative of atan2 at zero speed."""
    finite = torch.isfinite(x) & torch.isfinite(y)
    near_zero = finite & ((x.square() + y.square()) <= float(eps) ** 2)
    safe_x = torch.where(near_zero, torch.ones_like(x), x)
    safe_y = torch.where(near_zero, torch.zeros_like(y), y)
    return torch.atan2(safe_y, safe_x)


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
    """Pairwise ADE [B,A,M_pred,M_gt]."""
    pred_f = _safe_float(pred_traj)
    gt_f = _safe_float(gt_traj)
    return torch.linalg.norm(
        pred_f[:, :, :, None, :, :2] - gt_f[:, :, None, :, :, :2], dim=-1
    ).mean(dim=-1)


def _pairwise_ade_horizons(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    horizon_steps: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Compute all requested ADE horizons from one pairwise distance tensor.

    v16.4 materialized the same [B,A,M_pred,M_gt,T] displacement tensor four
    times (8 s plus 1/3/5 s).  The cumulative reduction below keeps the exact
    objective while removing three large repeated kernels and allocations.
    """
    pred_f = _safe_float(pred_traj)
    gt_f = _safe_float(gt_traj)
    distance = torch.linalg.norm(
        pred_f[:, :, :, None, :, :2] - gt_f[:, :, None, :, :, :2], dim=-1
    )
    cumulative = distance.cumsum(dim=-1)
    total_steps = int(distance.shape[-1])
    out: dict[int, torch.Tensor] = {}
    for requested in sorted(set(int(x) for x in horizon_steps)):
        steps = max(1, min(requested, total_steps))
        out[requested] = cumulative[..., steps - 1] / float(steps)
    return out


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


def _mode_source_tensor(
    pred_mode_source: torch.Tensor | None,
    pairwise_ade: torch.Tensor,
) -> torch.Tensor | None:
    if pred_mode_source is None:
        return None
    src = pred_mode_source.long().to(pairwise_ade.device)
    if src.ndim == 1:
        src = src[None, None, :].expand(pairwise_ade.shape[0], pairwise_ade.shape[1], -1)
    elif src.ndim == 2:
        src = src[:, None, :].expand(-1, pairwise_ade.shape[1], -1)
    if src.shape != pairwise_ade.shape[:3]:
        raise ValueError(
            f"predicted mode_source shape {tuple(src.shape)} incompatible with pairwise ADE {tuple(pairwise_ade.shape)}"
        )
    return src


def _typed_pairwise_ade(
    pairwise_ade: torch.Tensor,
    gt_source: torch.Tensor,
    pred_mode_source: torch.Tensor | None,
) -> torch.Tensor:
    """Mask cross-source assignments while preserving permutation invariance within a source.

    Stable source identities are required by root-indexed transport.  v13 used a
    global hard nearest-neighbour assignment, so one mode could explain OBS, NEU
    and PRIO roots simultaneously.  v14 only permits OBS->OBS, NEU->NEU and
    PRIO->PRIO matching; PAD/unknown targets retain the untyped fallback.
    """
    psrc = _mode_source_tensor(pred_mode_source, pairwise_ade)
    if psrc is None:
        return pairwise_ade
    gsrc = torch.nan_to_num(
        gt_source.float(), nan=float(NaturalSource.PAD),
        posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
    ).long().clamp(0, int(NaturalSource.PAD))
    allowed = psrc[..., :, None] == gsrc[..., None, :]
    allowed = allowed | (gsrc[..., None, :] == int(NaturalSource.PAD))
    large = torch.full_like(pairwise_ade, 1.0e4)
    return torch.where(allowed, pairwise_ade, large)


def _weighted_set_minade_from_pairwise(
    pairwise_ade: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    if not valid.any():
        return _zero_like_loss(ref)
    d_min = pairwise_ade.min(dim=2).values
    w = _nonnegative_weight(weight) * valid.float()
    if w.sum() <= 0:
        return _zero_like_loss(ref)
    return torch.where(valid.bool(), d_min * w, torch.zeros_like(d_min)).sum() / w.sum().clamp_min(1e-6)


def _natural_mixture_nll_from_pairwise(
    pairwise_ade: torch.Tensor,
    logits: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    ref: torch.Tensor,
    tau: float = 2.0,
) -> torch.Tensor:
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
    mix = F.softmax(_safe_float(logits), dim=-1)
    src_prob = F.softmax(_safe_float(source_logits), dim=-1)
    pred_src = (mix.unsqueeze(-1) * src_prob).sum(dim=2).clamp_min(1e-8)
    target = torch.zeros(*gt_source.shape[:2], source_count, device=logits.device, dtype=pred_src.dtype)
    src = torch.nan_to_num(
        gt_source.float(), nan=float(NaturalSource.PAD),
        posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
    ).long().clamp(0, source_count - 1)
    src_weight = _nonnegative_weight(weight).to(target.dtype) * valid.to(target.dtype)
    target.scatter_add_(-1, src, src_weight)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    agent_mask = valid.any(dim=-1) & (src_weight.sum(dim=-1) > 0)
    ce = -(target * torch.log(pred_src)).sum(dim=-1)
    return masked_mean(ce, agent_mask)


def _matched_natural_semantic_losses(
    typed_pairwise_ade: torch.Tensor,
    source_logits: torch.Tensor | None,
    priority_logits: torch.Tensor | None,
    gt_source: torch.Tensor,
    gt_priority: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not valid.any():
        z = _zero_like_loss(ref)
        return z, z
    with torch.no_grad():
        assignment = typed_pairwise_ade.argmin(dim=2)
    w = _nonnegative_weight(weight) * valid.float()
    denom = w.sum().clamp_min(1e-6)

    if source_logits is not None:
        src = torch.nan_to_num(
            gt_source.float(), nan=float(NaturalSource.PAD),
            posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
        ).long().clamp(0, int(source_logits.shape[-1]) - 1)
        idx = assignment[..., None].expand(*assignment.shape, source_logits.shape[-1])
        matched_src = torch.gather(source_logits, dim=2, index=idx)
        src_ce = F.cross_entropy(
            _safe_float(matched_src).reshape(-1, matched_src.shape[-1]),
            src.reshape(-1), reduction="none",
        ).reshape_as(src)
        source_loss = (src_ce * w).sum() / denom
    else:
        source_loss = _zero_like_loss(ref)

    if priority_logits is not None:
        matched_prio = torch.gather(priority_logits, dim=2, index=assignment)
        bce = F.binary_cross_entropy_with_logits(
            _safe_float(matched_prio), _binary_target(gt_priority), reduction="none"
        )
        priority_loss = (bce * w).sum() / denom
    else:
        priority_loss = _zero_like_loss(ref)
    return source_loss, priority_loss


def _set_minade_at_steps(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    gt_source: torch.Tensor,
    pred_mode_source: torch.Tensor | None,
    steps: int,
) -> torch.Tensor:
    steps = max(1, min(int(steps), int(pred_traj.shape[-2]), int(gt_traj.shape[-2])))
    pair = _pairwise_ade(pred_traj[..., :steps, :], gt_traj[..., :steps, :])
    pair = _typed_pairwise_ade(pair, gt_source, pred_mode_source)
    return _weighted_set_minade_from_pairwise(pair, valid, weight, pred_traj)


def _branch_minade_from_pairwise(
    typed_pairwise_ade: torch.Tensor,
    valid: torch.Tensor,
    source: torch.Tensor,
    branch: int,
    ref: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    branch_gt = valid & (source == int(branch))
    if not branch_gt.any():
        return _zero_like_loss(ref)
    d_min = typed_pairwise_ade.min(dim=2).values
    if weight is None:
        return masked_mean(d_min, branch_gt)
    return weighted_masked_mean(d_min, branch_gt, weight)


def _natural_residual_effectiveness_losses(
    pred: dict[str, torch.Tensor],
    gt_traj: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    gt_source: torch.Tensor,
    pred_mode_source: torch.Tensor | None,
    learned_pairwise: torch.Tensor,
    cfg: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Require residual learning to improve OBS without erasing causal priors."""
    z = _zero_like_loss(pred["traj"])
    if "base_traj" not in pred:
        return {"obs_shortfall": z, "neutral_degradation": z, "priority_degradation": z,
                "obs_gain": z, "neutral_gain": z, "priority_gain": z}
    base_pair = _typed_pairwise_ade(_pairwise_ade(pred["base_traj"], gt_traj), gt_source, pred_mode_source)
    learned_best = learned_pairwise.min(dim=2).values
    base_best = base_pair.min(dim=2).values
    delta = base_best - learned_best  # positive means the learned residual helped

    def wmean(branch: int, values: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        m = valid & (gt_source == int(branch))
        if extra is not None:
            m = m & extra
        return weighted_masked_mean(values, m, weight) if m.any() else z

    obs_difficult = base_best >= float(cfg.get("natural_obs_gain_min_base_ade_m", 0.50))
    obs_margin = float(cfg.get("natural_obs_gain_margin_m", 0.05))
    preserve_tol = float(cfg.get("natural_prior_degradation_tolerance_m", 0.10))
    obs_shortfall = wmean(int(NaturalSource.OBS), torch.relu(obs_margin - delta), obs_difficult)
    neu_degrade = wmean(int(NaturalSource.NEU), torch.relu(-delta - preserve_tol))
    prio_degrade = wmean(int(NaturalSource.PRIO), torch.relu(-delta - preserve_tol))
    return {
        "obs_shortfall": obs_shortfall,
        "neutral_degradation": neu_degrade,
        "priority_degradation": prio_degrade,
        "obs_gain": wmean(int(NaturalSource.OBS), delta),
        "neutral_gain": wmean(int(NaturalSource.NEU), delta),
        "priority_gain": wmean(int(NaturalSource.PRIO), delta),
    }


def _natural_kinematic_consistency_losses(
    pred: dict[str, torch.Tensor], critical_mask: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    traj = _safe_float(pred["traj"])
    z = _zero_like_loss(traj)
    if traj.shape[-2] < 2:
        return z, z, z
    dt = max(float(dt), 1e-4)
    pos = traj[..., 0:2]
    vel = traj[..., 3:5]
    fd_vel = (pos[..., 1:, :] - pos[..., :-1, :]) / dt
    midpoint_vel = 0.5 * (vel[..., 1:, :] + vel[..., :-1, :])
    mode_time_mask = critical_mask[:, :, None, None].expand_as(fd_vel[..., 0])
    velocity_loss = masked_mean(torch.linalg.norm(fd_vel - midpoint_vel, dim=-1), mode_time_mask)

    speed = torch.linalg.norm(vel[..., 1:, :], dim=-1)
    velocity_yaw = _safe_velocity_atan2(vel[..., 1:, 1], vel[..., 1:, 0])
    yaw = traj[..., 1:, 2]
    yaw_error = torch.abs(torch.atan2(torch.sin(yaw - velocity_yaw), torch.cos(yaw - velocity_yaw)))
    yaw_mask = mode_time_mask & (speed > 0.5)
    yaw_loss = masked_mean(yaw_error, yaw_mask) if yaw_mask.any() else z

    if "controls" in pred and pred["controls"].shape[-2] >= 2:
        controls = _safe_float(pred["controls"])
        smooth = torch.linalg.norm(controls[..., 1:, :] - controls[..., :-1, :], dim=-1)
        control_loss = masked_mean(smooth, critical_mask[:, :, None, None].expand_as(smooth))
    else:
        control_loss = z
    return velocity_loss, yaw_loss, control_loss


def _natural_residual_trust_region_losses(
    pred: dict[str, torch.Tensor],
    critical_mask: torch.Tensor,
    soft_ratio: float,
    *,
    mode_temperature: float = 1.0,
    probability_floor: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probability-mass-aware, multi-horizon root-identity regularization.

    The v16.4 endpoint ball treated every root identically and hard-clipped OBS
    modes that needed a larger physically plausible deviation.  COWP's OPR is a
    retained *probability mass* statement, so identity protection should focus on
    probability-carrying roots over the whole horizon.  The decoder separately
    enforces a wider emergency envelope for every mode.

    Mode probabilities are detached in this regularizer.  The mixture NLL still
    learns the logits, but the network cannot evade the geometric constraint by
    simply assigning zero probability to a violating trajectory.
    """
    traj = _safe_float(pred["traj"])
    z = _zero_like_loss(traj)
    ratio = pred.get("raw_residual_soft_path_ratio")
    logits = pred.get("logits")
    if ratio is None or logits is None:
        return z, z, z
    ratio_f = _safe_float(ratio)
    logits_f = _safe_float(logits)
    if ratio_f.shape != logits_f.shape:
        raise ValueError(
            f"raw_residual_soft_path_ratio shape {tuple(ratio_f.shape)} must match logits {tuple(logits_f.shape)}"
        )
    temperature = max(float(mode_temperature), 1.0e-3)
    prob = F.softmax(logits_f / temperature, dim=-1).detach()
    floor = min(max(float(probability_floor), 0.0), 0.25)
    if floor > 0.0:
        uniform = torch.full_like(prob, 1.0 / max(int(prob.shape[-1]), 1))
        prob = (1.0 - floor) * prob + floor * uniform
    mode_mask = critical_mask[:, :, None].expand_as(ratio_f)
    soft_ratio = min(max(float(soft_ratio), 0.0), 0.99)
    excess = torch.relu((ratio_f - soft_ratio) / max(1.0 - soft_ratio, 1.0e-3))
    penalty = weighted_masked_mean(excess.square(), mode_mask, prob)
    mean_ratio = weighted_masked_mean(ratio_f, mode_mask, prob)
    saturation = weighted_masked_mean((ratio_f >= 1.0).float(), mode_mask, prob)
    return penalty, mean_ratio, saturation

def _natural_mode_usage_loss(
    typed_pairwise: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor,
    gt_source: torch.Tensor, pred_mode_source: torch.Tensor | None, tau: float, ref: torch.Tensor
) -> torch.Tensor:
    psrc = _mode_source_tensor(pred_mode_source, typed_pairwise)
    if psrc is None:
        return _zero_like_loss(ref)
    total = _zero_like_loss(ref)
    branches = 0
    for branch in (int(NaturalSource.OBS), int(NaturalSource.NEU), int(NaturalSource.PRIO)):
        # ``psrc`` is [B,A,M].  v16 accidentally indexed ``[..., 0]`` here,
        # collapsing the mode axis and producing a [B,A] mask.  The error was
        # hidden by unit tests with A=1, but fails on real batches (A=6, M=24).
        mode_mask = psrc == branch
        eligible_modes = mode_mask.any(dim=0).any(dim=0)  # [M], compatible with older PyTorch
        n_modes = int(eligible_modes.sum().item())
        root_mask = valid & (gt_source == branch)
        if n_modes <= 1 or not bool(root_mask.any()):
            continue
        logits = -_safe_float(typed_pairwise) / max(float(tau), 1e-3)
        logits = torch.where(mode_mask[..., None], logits, torch.full_like(logits, -1.0e9))
        assignment = F.softmax(logits, dim=2)
        root_w = _nonnegative_weight(weight) * root_mask.float()
        mass = (assignment * root_w[:, :, None, :]).sum(dim=(0, 1, 3))
        mass = mass[eligible_modes].clamp_min(1e-12)
        prob = mass / mass.sum().clamp_min(1e-12)
        entropy = -(prob * prob.log()).sum()
        total = total + (1.0 - entropy / torch.log(prob.new_tensor(float(n_modes))))
        branches += 1
    return total / max(branches, 1)


def _diversity_loss(
    pred_traj: torch.Tensor,
    crit_mask: torch.Tensor,
    tau: float = 4.0,
    temporal_stride: int = 4,
) -> torch.Tensor:
    B, A, M = pred_traj.shape[:3]
    if M <= 1 or not crit_mask.any():
        return _zero_like_loss(pred_traj)
    stride = max(int(temporal_stride), 1)
    xy = _safe_float(pred_traj)[..., ::stride, :2]
    d = torch.linalg.norm(
        xy[:, :, :, None, :, :] - xy[:, :, None, :, :, :], dim=-1
    ).mean(dim=-1)
    eye = torch.eye(M, device=pred_traj.device, dtype=torch.bool)[None, None]
    pair_mask = crit_mask[:, :, None, None] & ~eye
    collapse = torch.exp(-d / max(float(tau), 1e-6))
    return masked_mean(collapse, pair_mask)


def natural_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Typed, source-restricted supervision for the natural option basis."""
    valid = batch["cowp/natural/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    mask = valid & crit[:, :, None]
    pred_mode_source = pred.get("mode_source")

    if mask.any():
        gt_traj = _safe_float(batch["cowp/natural/traj"])
        gt_source = torch.nan_to_num(
            batch["cowp/natural/source"].float(), nan=float(NaturalSource.PAD),
            posinf=float(NaturalSource.PAD), neginf=float(NaturalSource.PAD),
        ).long().clamp(0, int(NaturalSource.PAD))
        nat_weight = _nonnegative_weight(batch["cowp/natural/weight"])
        dt = float(weights.get("natural_dt", 0.1))
        total_steps = int(min(pred["traj"].shape[-2], gt_traj.shape[-2]))
        h1_steps = max(1, round(1.0 / dt))
        h3_steps = max(1, round(3.0 / dt))
        h5_steps = max(1, round(5.0 / dt))
        pairwise_by_horizon = _pairwise_ade_horizons(
            pred["traj"], gt_traj, (total_steps, h1_steps, h3_steps, h5_steps)
        )
        pairwise_ade = pairwise_by_horizon[total_steps]
        typed_pairwise = _typed_pairwise_ade(pairwise_ade, gt_source, pred_mode_source)

        # Untyped coverage is retained as a diagnostic; the optimized objective is
        # typed because RIOT needs source-stable roots, not merely any close curve.
        untyped_traj = _weighted_set_minade_from_pairwise(pairwise_ade, mask, nat_weight, pred["traj"])
        traj = _weighted_set_minade_from_pairwise(typed_pairwise, mask, nat_weight, pred["traj"])
        logits = pred["logits"]
        mode = _natural_mixture_nll_from_pairwise(
            typed_pairwise, logits, mask, nat_weight, pred["traj"],
            tau=float(weights.get("natural_mode_tau_m", 2.0)),
        )
        source_ce, priority_loss = _matched_natural_semantic_losses(
            typed_pairwise,
            pred.get("source_logits"),
            pred.get("priority_logits"),
            gt_source,
            batch["cowp/natural/priority_preserved"],
            mask,
            nat_weight,
            pred["traj"],
        )
        if "source_logits" in pred:
            source_distribution = _natural_source_distribution_loss(
                logits, pred["source_logits"], gt_source, mask, nat_weight,
                int(pred["source_logits"].shape[-1]),
            )
        else:
            source_distribution = _zero_like_loss(pred["traj"])

        obs_minade = _branch_minade_from_pairwise(typed_pairwise, mask, gt_source, int(NaturalSource.OBS), pred["traj"], nat_weight)
        neu_minade = _branch_minade_from_pairwise(typed_pairwise, mask, gt_source, int(NaturalSource.NEU), pred["traj"], nat_weight)
        prio_minade = _branch_minade_from_pairwise(typed_pairwise, mask, gt_source, int(NaturalSource.PRIO), pred["traj"], nat_weight)
        w_obs = float(weights.get("obs_prediction", 1.0))
        w_neu = float(weights.get("neutral", 0.5))
        w_prio = float(weights.get("priority_rule", 0.5))
        branch_minade = (w_obs * obs_minade + w_neu * neu_minade + w_prio * prio_minade) / max(w_obs + w_neu + w_prio, 1e-6)

        neu_gt_mask = mask & (gt_source == int(NaturalSource.NEU))
        psrc = _mode_source_tensor(pred_mode_source, pairwise_ade)
        if neu_gt_mask.any() and psrc is not None:
            typed_neu = psrc == int(NaturalSource.NEU)
            masked_logits = torch.where(typed_neu, _safe_float(logits), torch.full_like(logits.float(), -1.0e9))
            prob_neu = F.softmax(masked_logits, dim=-1) * typed_neu.float() * crit[:, :, None].float()
            pred_mu = (_safe_float(pred["traj"]) * prob_neu[..., None, None]).sum(dim=2) / prob_neu.sum(dim=2).clamp_min(1e-6)[..., None, None]
            gt_w = nat_weight * neu_gt_mask.float()
            gt_mu = (gt_traj * gt_w[..., None, None]).sum(dim=2) / gt_w.sum(dim=2).clamp_min(1e-6)[..., None, None]
            neu_cons = masked_mean(_trajectory_ade(pred_mu, gt_mu), neu_gt_mask.any(dim=-1))
        else:
            neu_cons = _zero_like_loss(pred["traj"])

        if float(weights.get("diversity_loss", weights.get("diversity", 0.05))) > 0.0:
            div = _diversity_loss(
                pred["traj"], crit & mask.any(dim=-1),
                tau=float(weights.get("natural_diversity_tau", 4.0)),
                temporal_stride=int(weights.get("natural_diversity_temporal_stride", 4)),
            )
        else:
            div = _zero_like_loss(pred["traj"])
        if "base_traj" in pred:
            base_per_mode = _trajectory_ade(pred["traj"], pred["base_traj"])
            mode_valid = (crit & mask.any(dim=-1))[:, :, None]
            psrc_for_reg = _mode_source_tensor(pred_mode_source, pairwise_ade)
            if psrc_for_reg is not None:
                base_dev_obs = masked_mean(
                    base_per_mode, mode_valid & (psrc_for_reg == int(NaturalSource.OBS))
                )
                base_dev_neu = masked_mean(
                    base_per_mode, mode_valid & (psrc_for_reg == int(NaturalSource.NEU))
                )
                base_dev_prio = masked_mean(
                    base_per_mode, mode_valid & (psrc_for_reg == int(NaturalSource.PRIO))
                )
            else:
                base_dev_obs = base_dev_neu = base_dev_prio = masked_mean(base_per_mode, mode_valid)
            base_dev = (base_dev_obs + base_dev_neu + base_dev_prio) / 3.0
        else:
            base_dev = _zero_like_loss(pred["traj"])
            base_dev_obs = base_dev_neu = base_dev_prio = base_dev
        if "residual" in pred:
            residual_l2 = masked_mean(
                _safe_float(pred["residual"])[..., :5].square().mean(dim=(-1, -2)),
                (crit & mask.any(dim=-1))[:, :, None],
            )
        else:
            residual_l2 = _zero_like_loss(pred["traj"])

        effectiveness_enabled = bool(weights.get("natural_compute_effectiveness_metrics", True)) or any(
            float(weights.get(k, 0.0)) > 0.0
            for k in (
                "natural_obs_gain", "natural_neutral_preservation",
                "natural_priority_preservation",
            )
        )
        if effectiveness_enabled:
            effectiveness = _natural_residual_effectiveness_losses(
                pred, gt_traj, mask, nat_weight, gt_source, pred_mode_source, typed_pairwise, weights
            )
        else:
            effectiveness = {k: _zero_like_loss(pred["traj"]) for k in (
                "obs_shortfall", "neutral_degradation", "priority_degradation",
                "obs_gain", "neutral_gain", "priority_gain",
            )}

        kinematic_enabled = any(
            float(weights.get(k, 0.0)) > 0.0
            for k in ("natural_kinematic_velocity", "natural_kinematic_yaw", "natural_control_smoothness")
        )
        if kinematic_enabled:
            kin_velocity, kin_yaw, control_smoothness = _natural_kinematic_consistency_losses(
                pred, crit & mask.any(dim=-1), dt
            )
        else:
            kin_velocity = kin_yaw = control_smoothness = _zero_like_loss(pred["traj"])

        if float(weights.get("natural_mode_usage", 0.0)) > 0.0:
            mode_usage = _natural_mode_usage_loss(
                typed_pairwise, mask, nat_weight, gt_source, pred_mode_source,
                tau=float(weights.get("natural_mode_usage_tau_m", 1.5)), ref=pred["traj"],
            )
        else:
            mode_usage = _zero_like_loss(pred["traj"])

        trust_region, trust_ratio, trust_saturation = _natural_residual_trust_region_losses(
            pred,
            crit & mask.any(dim=-1),
            soft_ratio=float(weights.get("natural_residual_trust_soft_ratio", 0.75)),
            mode_temperature=float(weights.get("natural_residual_trust_mode_temperature", 1.0)),
            probability_floor=float(weights.get("natural_residual_trust_probability_floor", 0.02)),
        )

        h1_pair = _typed_pairwise_ade(pairwise_by_horizon[h1_steps], gt_source, pred_mode_source)
        h3_pair = _typed_pairwise_ade(pairwise_by_horizon[h3_steps], gt_source, pred_mode_source)
        h5_pair = _typed_pairwise_ade(pairwise_by_horizon[h5_steps], gt_source, pred_mode_source)
        h1 = _weighted_set_minade_from_pairwise(h1_pair, mask, nat_weight, pred["traj"])
        h3 = _weighted_set_minade_from_pairwise(h3_pair, mask, nat_weight, pred["traj"])
        h5 = _weighted_set_minade_from_pairwise(h5_pair, mask, nat_weight, pred["traj"])
    else:
        traj = untyped_traj = mode = source_ce = source_distribution = priority_loss = branch_minade = neu_cons = div = _zero_like_loss(pred["traj"])
        obs_minade = neu_minade = prio_minade = base_dev = residual_l2 = _zero_like_loss(pred["traj"])
        base_dev_obs = base_dev_neu = base_dev_prio = base_dev
        h1 = h3 = h5 = _zero_like_loss(pred["traj"])
        kin_velocity = kin_yaw = control_smoothness = mode_usage = _zero_like_loss(pred["traj"])
        trust_region = trust_ratio = trust_saturation = _zero_like_loss(pred["traj"])
        effectiveness = {k: _zero_like_loss(pred["traj"]) for k in (
            "obs_shortfall", "neutral_degradation", "priority_degradation",
            "obs_gain", "neutral_gain", "priority_gain",
        )}

    total = (
        weights.get("natural_traj_l1", weights.get("obs_prediction", 1.0)) * traj
        + weights.get("natural_mode_ce", 0.5) * mode
        + weights.get("branch_source_ce", 0.4) * source_ce
        + weights.get("source_distribution_aux", 0.05) * source_distribution
        + weights.get("branch_minade", 0.7) * branch_minade
        + weights.get("priority_preservation", 0.4) * priority_loss
        + weights.get("neutral_consistency", 0.3) * neu_cons
        + weights.get("diversity_loss", weights.get("diversity", 0.05)) * div
        + weights.get("natural_short_1s", 0.25) * h1
        + weights.get("natural_short_3s", 0.15) * h3
        + weights.get("natural_short_5s", 0.10) * h5
        + weights.get("natural_base_deviation_obs", weights.get("natural_base_deviation", 0.05)) * base_dev_obs
        + weights.get("natural_base_deviation_neu", weights.get("natural_base_deviation", 0.05)) * base_dev_neu
        + weights.get("natural_base_deviation_prio", weights.get("natural_base_deviation", 0.05)) * base_dev_prio
        + weights.get("natural_residual_l2", 0.001) * residual_l2
        + weights.get("natural_obs_gain", 0.40) * effectiveness["obs_shortfall"]
        + weights.get("natural_neutral_preservation", 0.15) * effectiveness["neutral_degradation"]
        + weights.get("natural_priority_preservation", 0.15) * effectiveness["priority_degradation"]
        + weights.get("natural_kinematic_velocity", 0.10) * kin_velocity
        + weights.get("natural_kinematic_yaw", 0.05) * kin_yaw
        + weights.get("natural_control_smoothness", 0.01) * control_smoothness
        + weights.get("natural_mode_usage", 0.02) * mode_usage
        + weights.get("natural_residual_trust_region", 0.0) * trust_region
    )
    return {
        "loss": total,
        "traj": traj,
        "untyped_traj": untyped_traj,
        "mode": mode,
        "source": source_ce,
        "source_distribution": source_distribution,
        "priority": priority_loss,
        "branch_minade": branch_minade,
        "obs_minade": obs_minade,
        "neutral_minade": neu_minade,
        "prio_minade": prio_minade,
        "neutral_consistency": neu_cons,
        "diversity": div,
        "base_deviation": base_dev,
        "base_deviation_obs": base_dev_obs,
        "base_deviation_neu": base_dev_neu,
        "base_deviation_prio": base_dev_prio,
        "residual_l2": residual_l2,
        "residual_obs_shortfall": effectiveness["obs_shortfall"],
        "residual_neutral_degradation": effectiveness["neutral_degradation"],
        "residual_priority_degradation": effectiveness["priority_degradation"],
        "residual_obs_gain": effectiveness["obs_gain"],
        "residual_neutral_gain": effectiveness["neutral_gain"],
        "residual_priority_gain": effectiveness["priority_gain"],
        "kinematic_velocity": kin_velocity,
        "kinematic_yaw": kin_yaw,
        "control_smoothness": control_smoothness,
        "mode_usage": mode_usage,
        "residual_trust_region": trust_region,
        "residual_budget_ratio": trust_ratio,
        "residual_budget_saturation": trust_saturation,
        "minade_1s": h1,
        "minade_3s": h3,
        "minade_5s": h5,
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
    explanation_mask = pos_mask
    if "cowp/witness/explanation_valid" in batch:
        explanation_mask = explanation_mask & batch["cowp/witness/explanation_valid"].bool()

    if explanation_mask.any():
        token_target = torch.nan_to_num(
            batch["cowp/witness/token"].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).long().clamp(0, pred["token_logits"].shape[-1] - 1)
        token = F.cross_entropy(
            _safe_float(pred["token_logits"])[explanation_mask],
            token_target[explanation_mask],
            reduction="mean",
        )
        interval = F.smooth_l1_loss(
            _safe_float(pred["conflict_interval"])[explanation_mask],
            _safe_float(batch["cowp/witness/conflict_interval"])[explanation_mask],
            reduction="mean",
        )
    else:
        token = _zero_like_loss(pred["exist_logits"])
        interval = _zero_like_loss(pred["exist_logits"])

    if pos_mask.any():
        burden = F.smooth_l1_loss(
            _safe_float(pred["burden_total"])[pos_mask],
            _safe_float(batch["cowp/witness/burden_total"])[pos_mask],
            reduction="mean",
        )
        if "cowp/witness/burden_components" in batch and "burden_components" in pred:
            comps = F.smooth_l1_loss(
                _safe_float(pred["burden_components"])[pos_mask],
                _safe_float(batch["cowp/witness/burden_components"])[pos_mask],
                reduction="mean",
            )
        else:
            comps = _zero_like_loss(pred["exist_logits"])
    else:
        burden = _zero_like_loss(pred["exist_logits"])
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
    burden = masked_mean(
        torch.abs(_safe_float(pred["min_safe_burden"]).clamp(0.0, 2.0) - burden_target),
        pair,
    )
    tail_target = batch.get("cowp/witness/tail_burden_excess")
    if tail_target is not None and "tail_burden_excess" in pred:
        tail_burden = masked_mean(
            torch.abs(
                _safe_float(pred["tail_burden_excess"]).clamp(0.0, 2.0)
                - _safe_float(tail_target).clamp(0.0, 2.0)
            ),
            pair,
        )
    else:
        tail_burden = _zero_like_loss(pred)

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
        + float(weights.get("set_transport_tail_burden", 1.0)) * tail_burden
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
        "tail_burden": tail_burden, "conflict": conflict, "source": source, "response": response,
        "mode_conflict": mode_conflict, "mode_retain": mode_retain,
        "mode_uncertainty": mode_uncertainty, "mode_recovery": mode_recovery,
        "response_root_aux": response_root_aux, "root_recovery": root_recovery,
        "candidate_budget": candidate_budget,
        "candidate_budget_coverage": candidate_budget_coverage.detach(),
        "candidate_budget_ncf_rate": candidate_budget_ncf_rate.detach(),
        "candidate_budget_false_safe_rate": candidate_budget_false_safe_rate.detach(),
    }
