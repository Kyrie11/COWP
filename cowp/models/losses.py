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
    exist = pair_mined_focal_bce_with_logits(
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
    total = (
        weights.get("witness_exist", 2.0) * exist
        + weights.get("witness_token", 1.0) * token
        + weights.get("witness_burden", 0.5) * burden
        + weights.get("witness_burden_components", 0.25) * comps
        + weights.get("witness_conflict", 0.3) * interval
        + weights.get("witness_opr", 0.5) * (opr + ci)
    )
    return {"loss": total, "exist": exist, "token": token, "burden": burden, "components": comps, "interval": interval, "opr": opr, "ci": ci}


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
        + comps_w * loss_comps
        + traj_w * loss_traj
    )
    return {"loss": total, "safe": loss_safe, "low": loss_low, "burden": loss_b, "components": loss_comps, "traj": loss_traj}


def candidate_classification_loss(pred_scores: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Candidate-level auxiliary supervision for learned planner quality."""
    mask = batch["cowp/candidates/valid"].bool()
    ncf = _binary_target(batch["cowp/candidates/noncoercive_feasible"])
    false_safe = _binary_target(batch["cowp/candidates/false_safe"])
    pred_scores = _safe_float(pred_scores)
    # Lower planner score is better, so supervise -score as NCF logit and score as false-safe logit.
    ncf_loss = F.binary_cross_entropy_with_logits(-pred_scores, ncf, reduction="none")
    fs_loss = F.binary_cross_entropy_with_logits(pred_scores, false_safe, reduction="none")
    loss_ncf = masked_mean(ncf_loss, mask)
    loss_fs = masked_mean(fs_loss, mask)
    total = weights.get("candidate_ncf_cls", 1.0) * loss_ncf + weights.get("candidate_false_safe_cls", 0.5) * loss_fs
    return {"loss": total, "ncf": loss_ncf, "false_safe": loss_fs}


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
        logits = torch.where(mask[b], -scores[b], torch.full_like(scores[b], -1e9))
        losses.append(F.cross_entropy(logits.unsqueeze(0), tgt[:1]))
    return torch.stack(losses).mean() if losses else _zero_like_loss(scores)


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
    prob = F.softmax(torch.where(mask, -scores, torch.full_like(scores, -1e9)), dim=-1)
    return (prob * cost).sum(dim=-1).mean()


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
