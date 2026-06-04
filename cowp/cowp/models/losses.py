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
    else:
        token = pred["exist_logits"].sum() * 0.0
        burden = pred["exist_logits"].sum() * 0.0
        interval = pred["exist_logits"].sum() * 0.0
    opr = masked_mean(torch.abs(pred["opr"] - batch["cowp/witness/opr"].float()), pair_mask)
    ci = masked_mean(torch.abs(pred["c_i"] - batch["cowp/witness/c_i"].float()), pair_mask)
    total = weights.get("witness_exist", 2.0) * exist + weights.get("witness_token", 1.0) * token + weights.get("witness_burden", 0.5) * burden + weights.get("witness_conflict", 0.3) * interval + weights.get("witness_opr", 0.5) * (opr + ci)
    return {"loss": total, "exist": exist, "token": token, "burden": burden, "interval": interval, "opr": opr, "ci": ci}


def response_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    mask = batch["cowp/response/valid"].bool()
    safe = F.binary_cross_entropy_with_logits(pred["safe_logits"], batch["cowp/response/is_safe"].float(), reduction="none")
    low = F.binary_cross_entropy_with_logits(pred["low_logits"], batch["cowp/response/is_low_burden"].float(), reduction="none")
    b = torch.abs(pred["burden_total"] - batch["cowp/response/burden_total"].float())
    loss_safe = masked_mean(safe, mask)
    loss_low = masked_mean(low, mask)
    loss_b = masked_mean(b, mask)
    total = weights.get("response_safe_bce", 1.0) * loss_safe + weights.get("response_low_bce", 1.0) * loss_low + weights.get("response_burden_l1", 0.5) * loss_b
    return {"loss": total, "safe": loss_safe, "low": loss_low, "burden": loss_b}


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
