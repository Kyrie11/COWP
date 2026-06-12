from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.float()
    return (value * mask_f).sum() / mask_f.sum().clamp_min(eps)


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    w = torch.where(target > 0.5, alpha, 1 - alpha) * (1 - pt).pow(gamma)
    return masked_mean(bce * w, mask)


def natural_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    """Supervise the learned natural-alternative branch.

    Label construction stores an ordered set of plausible uncoerced futures and a
    normalized mixture weight for every critical agent.  We supervise both the
    modal trajectories and the mixture logits.  This makes the branch trainable
    instead of being an unused decoder during the witness/planner stages.
    """
    valid = batch["cowp/natural/valid"].bool()
    crit = batch["cowp/critical/valid"].bool()
    mask = valid & crit[:, :, None]
    if mask.any():
        traj_l1 = torch.abs(pred["traj"] - batch["cowp/natural/traj"].float()).mean(dim=(-1, -2))
        traj = masked_mean(traj_l1, mask)
        logits = pred["logits"]
        target_w = batch["cowp/natural/weight"].float() * mask.float()
        target_w = target_w / target_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(logits, dim=-1)
        kl = -(target_w * logp).sum(dim=-1)
        mode = masked_mean(kl, crit & mask.any(dim=-1))
    else:
        traj = pred["traj"].sum() * 0.0
        mode = pred["logits"].sum() * 0.0
    total = weights.get("natural_traj_l1", weights.get("obs_prediction", 1.0)) * traj + weights.get("natural_mode_ce", weights.get("neutral", 0.5)) * mode
    return {"loss": total, "traj": traj, "mode": mode}


def witness_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    cand_mask = batch["cowp/candidates/valid"].bool()
    crit_mask = batch["cowp/critical/valid"].bool()
    pair_mask = cand_mask[:, :, None] & crit_mask[:, None, :]
    y = batch["cowp/witness/exists"].bool()
    exist = focal_bce_with_logits(pred["exist_logits"], y.float(), pair_mask)
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
