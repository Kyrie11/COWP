from __future__ import annotations

import torch
import torch.nn.functional as F

from cowp.core.constants import NaturalSource


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.float()
    return (value * mask_f).sum() / mask_f.sum().clamp_min(eps)


def _focal_bce_values(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    w = torch.where(target > 0.5, alpha, 1 - alpha) * (1 - pt).pow(gamma)
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
    """Focal BCE with pair-level positive / hard-negative mining.

    Scene-level oversampling helps put witness-positive scenes into a batch, but
    most candidate-agent pairs inside those scenes are still easy negatives.  The
    paper's witness objective is pair-level, so we keep positives and the hardest
    negatives per scene according to the current focal loss.
    """
    values = _focal_bce_values(logits, target.float(), gamma=gamma, alpha=alpha)
    keep = torch.zeros_like(mask, dtype=torch.bool)
    B = values.shape[0]
    flat_values = values.reshape(B, -1)
    flat_mask = mask.reshape(B, -1).bool()
    flat_target = target.reshape(B, -1).bool()
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
            return v.sum() * 0.0
    raise ValueError("prediction dict contains no tensors")


def _trajectory_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_traj[..., :2] - gt_traj[..., :2], dim=-1).mean(dim=-1)



def _weighted_set_minade(pred_traj: torch.Tensor, gt_traj: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unordered set supervision for natural alternatives.

    Natural alternatives are a weighted counterfactual set, not an ordered list.
    Penalizing slot-by-slot L1 makes training depend on construction order; this
    term asks every valid labelled alternative to be covered by some predicted
    mode, weighted by its probability mass.
    """
    if not valid.any():
        return pred_traj.sum() * 0.0
    d = torch.linalg.norm(pred_traj[:, :, :, None, :, :2] - gt_traj[:, :, None, :, :, :2], dim=-1).mean(dim=-1)
    d_min = d.min(dim=2).values
    w = weight.float() * valid.float()
    return (d_min * w).sum() / w.sum().clamp_min(1e-6)



def _natural_mixture_nll(pred_traj: torch.Tensor, logits: torch.Tensor, gt_traj: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """Order-invariant mixture supervision for natural alternatives.

    Each labelled alternative is explained by a soft log-sum over predicted modes,
    weighted by its counterfactual probability mass.  This avoids treating the
    construction order OBS/NEU/PRIO as a fixed decoder slot order.
    """
    if not valid.any():
        return pred_traj.sum() * 0.0
    d = torch.linalg.norm(pred_traj[:, :, :, None, :, :2] - gt_traj[:, :, None, :, :, :2], dim=-1).mean(dim=-1)  # [B,A,Mp,Mg]
    logp = F.log_softmax(logits, dim=-1)[:, :, :, None]
    log_cover = torch.logsumexp(logp - d / max(float(tau), 1e-6), dim=2)  # [B,A,Mg]
    w = weight.float() * valid.float()
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
        return logits.sum() * 0.0
    mix = F.softmax(logits, dim=-1)
    src_prob = F.softmax(source_logits, dim=-1)
    pred_src = (mix.unsqueeze(-1) * src_prob).sum(dim=2).clamp_min(1e-8)  # [B,A,S]
    pred_log = torch.log(pred_src)
    target = torch.zeros(*gt_source.shape[:2], source_count, device=logits.device, dtype=logits.dtype)
    src = gt_source.clamp(0, source_count - 1)
    target.scatter_add_(-1, src, weight.float() * valid.float())
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    agent_mask = valid.any(dim=-1)
    ce = -(target * pred_log).sum(dim=-1)
    return masked_mean(ce, agent_mask)


def _natural_priority_expectation_loss(logits: torch.Tensor, priority_logits: torch.Tensor, gt_priority: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0.0
    mix = F.softmax(logits, dim=-1)
    pred_p = (mix * torch.sigmoid(priority_logits)).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
    w = weight.float() * valid.float()
    tgt = (gt_priority.float() * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1e-6)
    agent_mask = valid.any(dim=-1)
    bce = F.binary_cross_entropy(pred_p, tgt, reduction="none")
    return masked_mean(bce, agent_mask)

def _branch_minade(pred_traj: torch.Tensor, gt_traj: torch.Tensor, valid: torch.Tensor, source: torch.Tensor, branch: int) -> torch.Tensor:
    branch_gt = valid & (source == int(branch))
    if not branch_gt.any():
        return pred_traj.sum() * 0.0
    # Pairwise ADE [B,A,M_pred,M_gt].  This is small for the configured K/A/M and
    # avoids binding the decoder to a fixed arbitrary label ordering.
    d = torch.linalg.norm(pred_traj[:, :, :, None, :, :2] - gt_traj[:, :, None, :, :, :2], dim=-1).mean(dim=-1)
    d_min = d.min(dim=2).values
    return masked_mean(d_min, branch_gt)


def _diversity_loss(pred_traj: torch.Tensor, crit_mask: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    B, A, M = pred_traj.shape[:3]
    if M <= 1 or not crit_mask.any():
        return pred_traj.sum() * 0.0
    xy = pred_traj[..., :2]
    d = torch.linalg.norm(xy[:, :, :, None, :, :] - xy[:, :, None, :, :, :], dim=-1).mean(dim=-1)
    eye = torch.eye(M, device=pred_traj.device, dtype=torch.bool)[None, None]
    pair_mask = crit_mask[:, :, None, None] & ~eye
    collapse = torch.exp(-d / max(float(tau), 1e-6))
    return masked_mean(collapse, pair_mask)


def natural_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Supervise natural alternatives with branch-aware losses.

    Terms implemented here mirror the paper appendix:
    branch source classification, branch-specific minADE, priority preservation,
    neutral-branch consistency, and diversity/coverage regularization.
    """
    valid = batch["cowp/natural/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    mask = valid & crit[:, :, None]
    if mask.any():
        gt_traj = batch["cowp/natural/traj"].float()
        gt_source = batch["cowp/natural/source"].long().clamp_min(0)
        # Set-valued minADE rather than slotwise L1: natural alternatives are
        # unordered observational / neutral / priority-preserving branches.
        traj = _weighted_set_minade(pred["traj"], gt_traj, mask, batch["cowp/natural/weight"].float())
        logits = pred["logits"]
        mode = _natural_mixture_nll(
            pred["traj"],
            logits,
            gt_traj,
            mask,
            batch["cowp/natural/weight"].float(),
            tau=float(weights.get("natural_mode_tau_m", 2.0)),
        )

        if "source_logits" in pred:
            source_ce = _natural_source_distribution_loss(
                logits,
                pred["source_logits"],
                gt_source,
                mask,
                batch["cowp/natural/weight"].float(),
                int(pred["source_logits"].shape[-1]),
            )
        else:
            source_ce = pred["traj"].sum() * 0.0
        if "priority_logits" in pred:
            priority_loss = _natural_priority_expectation_loss(
                logits,
                pred["priority_logits"],
                batch["cowp/natural/priority_preserved"].float(),
                mask,
                batch["cowp/natural/weight"].float(),
            )
        else:
            priority_loss = pred["traj"].sum() * 0.0

        obs_minade = _branch_minade(pred["traj"], gt_traj, mask, gt_source, int(NaturalSource.OBS))
        neu_minade = _branch_minade(pred["traj"], gt_traj, mask, gt_source, int(NaturalSource.NEU))
        prio_minade = _branch_minade(pred["traj"], gt_traj, mask, gt_source, int(NaturalSource.PRIO))
        branch_minade = (obs_minade + neu_minade + prio_minade) / 3.0

        # Neutral consistency compares source-probability weighted predicted neutral
        # mean with the label neutral mean for each critical agent.
        neu_gt_mask = mask & (gt_source == int(NaturalSource.NEU))
        if neu_gt_mask.any() and "source_logits" in pred:
            prob_neu = F.softmax(pred["source_logits"], dim=-1)[..., int(NaturalSource.NEU)] * crit[:, :, None].float()
            pred_mu = (pred["traj"] * prob_neu[..., None, None]).sum(dim=2) / prob_neu.sum(dim=2).clamp_min(1e-6)[..., None, None]
            gt_w = batch["cowp/natural/weight"].float() * neu_gt_mask.float()
            gt_mu = (gt_traj * gt_w[..., None, None]).sum(dim=2) / gt_w.sum(dim=2).clamp_min(1e-6)[..., None, None]
            agent_has_neu = neu_gt_mask.any(dim=-1)
            neu_cons = masked_mean(_trajectory_ade(pred_mu, gt_mu), agent_has_neu)
        else:
            neu_cons = pred["traj"].sum() * 0.0
        div = _diversity_loss(pred["traj"], crit & mask.any(dim=-1), tau=float(weights.get("natural_diversity_tau", 4.0)))
    else:
        traj = pred["traj"].sum() * 0.0
        mode = pred.get("logits", pred["traj"]).sum() * 0.0
        source_ce = priority_loss = branch_minade = neu_cons = div = pred["traj"].sum() * 0.0
        obs_minade = neu_minade = prio_minade = pred["traj"].sum() * 0.0

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
    y = batch["cowp/witness/exists"].bool()
    exist = pair_mined_focal_bce_with_logits(
        pred["exist_logits"],
        y.float(),
        pair_mask,
        gamma=float(weights.get("witness_focal_gamma", 2.0)),
        alpha=float(weights.get("witness_focal_alpha", 0.25)),
        max_pos_per_scene=int(weights.get("witness_mining_max_pos_per_scene", 16)),
        max_neg_per_scene=int(weights.get("witness_mining_max_neg_per_scene", 48)),
        neg_pos_ratio=int(weights.get("witness_mining_neg_pos_ratio", 3)),
        min_neg_per_scene=int(weights.get("witness_mining_min_neg_per_scene", 8)),
    )
    pos_mask = pair_mask & y
    if pos_mask.any():
        token = F.cross_entropy(pred["token_logits"][pos_mask], batch["cowp/witness/token"].long()[pos_mask], reduction="mean")
        burden = F.smooth_l1_loss(pred["burden_total"][pos_mask], batch["cowp/witness/burden_total"].float()[pos_mask], reduction="mean")
        interval = F.smooth_l1_loss(pred["conflict_interval"][pos_mask], batch["cowp/witness/conflict_interval"].float()[pos_mask], reduction="mean")
        if "cowp/witness/burden_components" in batch and "burden_components" in pred:
            comps = F.smooth_l1_loss(pred["burden_components"][pos_mask], batch["cowp/witness/burden_components"].float()[pos_mask], reduction="mean")
        else:
            comps = pred["exist_logits"].sum() * 0.0
    else:
        token = pred["exist_logits"].sum() * 0.0
        burden = pred["exist_logits"].sum() * 0.0
        interval = pred["exist_logits"].sum() * 0.0
        comps = pred["exist_logits"].sum() * 0.0
    opr = masked_mean(torch.abs(pred["opr"] - batch["cowp/witness/opr"].float()), pair_mask)
    ci = masked_mean(torch.abs(pred["c_i"] - batch["cowp/witness/c_i"].float()), pair_mask)
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
    # Response labels are candidate-critical-response tensors [B,K,A,R].  Critical
    # slots that are not visible in the WOMD input must be removed here as well;
    # natural_loss and witness_loss already consume cowp/critical/valid directly.
    if "cowp/critical/valid" in batch:
        mask = mask & batch["cowp/critical/valid"].bool()[:, None, :, None]
    safe = F.binary_cross_entropy_with_logits(pred["safe_logits"], batch["cowp/response/is_safe"].float(), reduction="none")
    low = F.binary_cross_entropy_with_logits(pred["low_logits"], batch["cowp/response/is_low_burden"].float(), reduction="none")
    b = torch.abs(pred["burden_total"] - batch["cowp/response/burden_total"].float())
    loss_safe = masked_mean(safe, mask)
    loss_low = masked_mean(low, mask)
    loss_b = masked_mean(b, mask)
    if "cowp/response/traj" in batch:
        traj_l1 = torch.abs(pred["traj"] - batch["cowp/response/traj"].float()).mean(dim=(-1, -2))
        loss_traj = masked_mean(traj_l1, mask)
    else:
        loss_traj = pred["safe_logits"].sum() * 0.0
    if "cowp/response/burden_components" in batch and "burden_components" in pred:
        comps_l1 = torch.abs(pred["burden_components"] - batch["cowp/response/burden_components"].float()).mean(dim=-1)
        loss_comps = masked_mean(comps_l1, mask)
    else:
        loss_comps = pred["safe_logits"].sum() * 0.0
    total = (
        weights.get("response_safe_bce", 1.0) * loss_safe
        + weights.get("response_low_bce", 1.0) * loss_low
        + weights.get("response_burden_l1", 0.5) * loss_b
        + weights.get("response_components_l1", 0.25) * loss_comps
        + weights.get("response_traj_l1", 0.1) * loss_traj
    )
    return {"loss": total, "safe": loss_safe, "low": loss_low, "burden": loss_b, "components": loss_comps, "traj": loss_traj}


def candidate_classification_loss(pred_scores: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Candidate-level auxiliary supervision for learned planner quality."""
    mask = batch["cowp/candidates/valid"].bool()
    ncf = batch["cowp/candidates/noncoercive_feasible"].float()
    false_safe = batch["cowp/candidates/false_safe"].float()
    # Lower planner score is better, so supervise -score as NCF logit and score as false-safe logit.
    ncf_loss = F.binary_cross_entropy_with_logits(-pred_scores, ncf, reduction="none")
    fs_loss = F.binary_cross_entropy_with_logits(pred_scores, false_safe, reduction="none")
    loss_ncf = masked_mean(ncf_loss, mask)
    loss_fs = masked_mean(fs_loss, mask)
    total = weights.get("candidate_ncf_cls", 1.0) * loss_ncf + weights.get("candidate_false_safe_cls", 0.5) * loss_fs
    return {"loss": total, "ncf": loss_ncf, "false_safe": loss_fs}




def planner_imitation_loss(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Imitation term using the logged ego candidate when available.

    Scores are minimized at inference, so ``-scores`` are used as selection logits.
    """
    mask = batch["cowp/candidates/valid"].bool()
    logged = batch.get("cowp/candidates/is_logged")
    if logged is None:
        return scores.sum() * 0.0
    target_mask = logged.bool() & mask
    losses = []
    for b in range(scores.shape[0]):
        tgt = torch.where(target_mask[b])[0]
        if tgt.numel() == 0:
            continue
        logits = torch.where(mask[b], -scores[b], torch.full_like(scores[b], -1e9))
        losses.append(F.cross_entropy(logits.unsqueeze(0), tgt[:1]))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def planner_outcome_loss(scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Optional rollout/outcome surrogate when Waymax candidate labels exist."""
    mask = batch["cowp/candidates/valid"].bool()
    collision = batch.get("waymax/candidate_collision")
    offroad = batch.get("waymax/candidate_offroad")
    logdiv = batch.get("waymax/candidate_log_divergence")
    if collision is None and offroad is None and logdiv is None:
        return scores.sum() * 0.0
    cost = torch.zeros_like(scores, dtype=torch.float32)
    if collision is not None:
        cost = cost + collision.float()
    if offroad is not None:
        cost = cost + offroad.float()
    if logdiv is not None:
        cost = cost + logdiv.float().clamp_min(0.0) / 10.0
    prob = F.softmax(torch.where(mask, -scores, torch.full_like(scores, -1e9)), dim=-1)
    return (prob * cost).sum(dim=-1).mean()

def planner_ranking_loss(scores: torch.Tensor, ncf: torch.Tensor, false_safe: torch.Tensor, cand_mask: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    losses = []
    B = scores.shape[0]
    for b in range(B):
        pos = torch.where(cand_mask[b] & ncf[b].bool())[0]
        neg = torch.where(cand_mask[b] & false_safe[b].bool())[0]
        if len(pos) and len(neg):
            # Lower score is better.
            losses.append(torch.relu(margin + scores[b, pos[:, None]] - scores[b, neg[None, :]]).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0
