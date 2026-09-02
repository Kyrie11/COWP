from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ocrap.algorithms.evidence_targets import (
    ComponentVetoTolerances,
    component_veto_margin_torch,
    component_veto_soft_target,
    component_veto_terms_torch,
)


def margin_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def frontier_normalize_signed_margin(
    margin: torch.Tensor, scale: float
) -> torch.Tensor:
    """Odd monotone compression that preserves the physical zero frontier.

    v48.40 DCFR uses this only for dense component-margin regression. BCE sign
    targets, measured hard vetoes, and deployment margins keep their original
    physical semantics. Large violations therefore cannot dominate regression
    capacity while ±tolerance examples retain high resolution.
    """
    s = max(float(scale), 1.0e-6)
    return s * torch.tanh(margin / s)




def _component_vector(
    raw: str | tuple[float, ...] | list[float] | None,
    *,
    ncomp: int,
    default: float,
    clamp_unit: bool = False,
) -> list[float]:
    if raw is None:
        values: list[float] = []
    elif isinstance(raw, str):
        text = raw.strip()
        values = [] if text.lower() in {"", "none", "null", "~"} else [
            float(x.strip()) for x in text.split(",") if x.strip()
        ]
    else:
        values = [float(x) for x in raw]
    if not values:
        values = [float(default)] * int(ncomp)
    if len(values) < int(ncomp):
        values.extend([float(default)] * (int(ncomp) - len(values)))
    values = values[: int(ncomp)]
    if clamp_unit:
        values = [min(1.0, max(0.0, x)) for x in values]
    return values


def component_margin_regression_targets(
    raw_margin: torch.Tensor,
    *,
    mode: str,
    target_scale: float,
    canonical_scales: str | tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    """Transform dense component margins without changing hard-veto semantics."""
    target_mode = str(mode or "raw").strip().lower()
    if target_mode == "raw":
        return raw_margin
    if target_mode == "frontier_tanh":
        return frontier_normalize_signed_margin(raw_margin, target_scale)
    if target_mode != "pooled_rms_linear":
        raise ValueError(f"unsupported ordinal_evidence_component_margin_target_mode={mode!r}")
    ncomp = int(raw_margin.shape[-1])
    values = _component_vector(canonical_scales, ncomp=ncomp, default=float("nan"))
    scales = raw_margin.new_tensor(values)
    if (not torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError(f"invalid component canonical scales: {values!r}")
    # v48.55 TCBC: linear, component-specific normalization.  It preserves
    # the exact zero crossing and all within-component ordering and does not
    # saturate large margins (unlike the historical v48.40 frontier_tanh).
    return float(target_scale) * raw_margin / scales.unsqueeze(0)


def anti_oracle_loss(pred_r_orc: torch.Tensor, pred_r_dep: torch.Tensor, teacher_artifact: torch.Tensor, delta_neg: float = 0.0) -> torch.Tensor:
    """Anti-oracle loss from Eq. (18): I_art * [R_dep - delta_neg]_+.

    The loss must push predicted deployable recoverability down on oracle
    artifacts.  Penalizing the oracle-deployability gap itself would instead
    encourage the model to close the gap, which is the opposite of the paper's
    anti-oracle objective.
    """
    del pred_r_orc  # retained in the signature for backwards compatibility
    return F.relu(pred_r_dep - float(delta_neg)).mul(teacher_artifact.float()).mean()


def artifact_gap_loss(pred_gap: torch.Tensor, teacher_gap: torch.Tensor, teacher_artifact: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Encourage a positive oracle-to-deployable gap on oracle artifacts.

    The main SmoothL1 losses regress R_orc and R_dep separately.  In practice,
    rare artifact cases can be underweighted, so this auxiliary ranking-style
    term directly teaches the model that artifacts should preserve an oracle
    headroom/deployable-headroom separation.
    """
    mask = teacher_artifact.float() > 0.5
    if not bool(mask.any()):
        return pred_gap.sum() * 0.0
    target = torch.clamp(teacher_gap.float(), min=float(margin))
    return F.relu(target[mask] - pred_gap[mask]).mean()


def deployability_classification_loss(pred_r_dep: torch.Tensor, teacher_r_dep: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Balanced binary supervision for calibrated deployability admission.

    Regression alone is weak near the admission threshold.  This term makes the
    sign/threshold of predicted deployable recovery learnable while keeping the
    numerical margin regression intact.
    """
    target = (teacher_r_dep.float() >= float(gamma)).float()
    logits = pred_r_dep.float() - float(gamma)
    pos = target > 0.5
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return F.binary_cross_entropy_with_logits(logits, target)
    weight = torch.where(pos, 0.5 / pos.float().mean().clamp_min(1e-6), 0.5 / neg.float().mean().clamp_min(1e-6))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def shared_option_q_regression_loss(pred_q: torch.Tensor, teacher_q: torch.Tensor, root_valid: torch.Tensor, option_valid: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 supervision for the shared recovery-option value Q(i,l).

    Earlier training supervised only aggregated R_dep/R_orc and raw margins.
    That lets the model learn that a prefix is less oracle-artifact-like, but it
    can still pick the wrong shared recovery option at execution time.  This
    auxiliary target directly teaches the option table used by the selector and
    by DRS evaluation.
    """
    mask = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = torch.clamp(torch.nan_to_num(teacher_q.detach(), nan=0.0, posinf=5.0, neginf=-5.0), min=-5.0, max=5.0)
    pred = torch.clamp(torch.nan_to_num(pred_q, nan=0.0, posinf=5.0, neginf=-5.0), min=-5.0, max=5.0)
    return F.smooth_l1_loss(pred[mask], target[mask])


def shared_option_admission_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Balanced BCE for whether each shared option is deployably admissible.

    This is the supervision missing when offline results show low ODG/FRA_cand
    but poor executed DRS: the model knows which prefix is less artifact-prone,
    but not which shared option is actually executable across compatible roots.
    """
    mask = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = (teacher_q.detach() >= float(gamma)).float()
    logits = torch.nan_to_num(pred_q - float(gamma), nan=-20.0, posinf=20.0, neginf=-20.0)
    weights = torch.clamp(root_probs.float(), min=0.0).unsqueeze(-1).expand_as(logits)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    # Balance positives and negatives so rare deployable options are not washed out.
    pos = (target > 0.5) & mask
    neg = (target <= 0.5) & mask
    if bool(pos.any()) and bool(neg.any()):
        pos_scale = 0.5 / pos.float().mean().clamp_min(1e-6)
        neg_scale = 0.5 / neg.float().mean().clamp_min(1e-6)
        weights = weights * torch.where(pos, pos_scale, torch.where(neg, neg_scale, torch.zeros_like(weights)))
    denom = weights.sum().clamp_min(1.0)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (loss * weights).sum() / denom


def _root_weights(root_probs: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(root_probs.float(), min=0.0) * root_valid.bool().float()
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def shared_option_success_regression_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Regress the probability that a *single shared* option succeeds.

    Executed DRS is not a per-root best-action metric.  At runtime the planner
    must pick one recovery option shared across all compatible roots.  This loss
    aggregates root mass first and supervises the option-level success vector
    that the selector actually needs.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    teacher_prob = (teacher_q.detach() >= float(gamma)).float()
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1)
    teacher_success = (teacher_prob * root_w * mask.float()).sum(dim=1)
    valid_opt = opt_mask & mask.any(dim=1)
    if not bool(valid_opt.any()):
        return pred_q.sum() * 0.0
    return F.smooth_l1_loss(pred_success[valid_opt], teacher_success[valid_opt])


def shared_option_success_bce_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Stronger BCE supervision for shared-option success probability.

    Smooth-L1 on success probability is often numerically small even when the
    ranking is poor.  This BCE target gives larger gradients when the model is
    over-confident about low-success shared recovery options, which was the
    failure mode observed in v8: predicted DRS proxy was high while executed DRS
    remained low.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    teacher_prob = (teacher_q.detach() >= float(gamma)).float()
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
    teacher_success = (teacher_prob * root_w * mask.float()).sum(dim=1).clamp(0.0, 1.0)
    valid_opt = opt_mask & mask.any(dim=1)
    if not bool(valid_opt.any()):
        return pred_q.sum() * 0.0
    # Upweight ambiguous/intervention-critical options where success is neither
    # trivially 0 nor trivially 1.
    t = teacher_success[valid_opt]
    p = pred_success[valid_opt]
    w = 1.0 + 2.0 * torch.minimum(t, 1.0 - t)
    return (F.binary_cross_entropy(p, t, reduction="none") * w).mean()


def best_shared_option_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """CE for the best globally shared recovery option.

    The old target used argmax_l Q(i,l) independently for each root i.  That is
    misaligned with deployability because one executed option must work across
    the compatible root set.  The teacher target is now the option maximizing
    root-mass deployable success, with a small value tie-breaker.
    """
    root_w = _root_weights(root_probs, root_valid).unsqueeze(-1)
    finite = torch.isfinite(teacher_q)
    opt_mask = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & opt_mask.unsqueeze(1) & finite
    if not bool(mask.any()) or not bool(opt_mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    teacher_success = ((teacher_q.detach() >= float(gamma)).float() * root_w * mask.float()).sum(dim=1)
    teacher_value = (torch.clamp(torch.nan_to_num(teacher_q.detach(), nan=-5.0, posinf=5.0, neginf=-5.0), -5.0, 5.0) * root_w * mask.float()).sum(dim=1)
    teacher_score = teacher_success + 0.05 * teacher_value
    teacher_score = torch.where(opt_mask, teacher_score, torch.full_like(teacher_score, -1.0e9))
    target = torch.argmax(teacher_score, dim=-1)

    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    pred_success = (pred_prob * root_w * mask.float()).sum(dim=1)
    pred_value = (torch.clamp(torch.nan_to_num(pred_q, nan=-5.0, posinf=5.0, neginf=-5.0), -5.0, 5.0) * root_w * mask.float()).sum(dim=1)
    logits = pred_success + 0.05 * pred_value
    logits = torch.where(opt_mask, logits, torch.full_like(logits, -1.0e4))
    valid_row = opt_mask.any(dim=-1) & mask.any(dim=(1, 2))
    if not bool(valid_row.any()):
        return pred_q.sum() * 0.0
    return F.cross_entropy(logits[valid_row], target[valid_row])




def recovery_conflict_pair_weights(
    teacher_margins: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.20,
    conflict_scale: float = 3.0,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Decision-importance weights for the physical observation-equivalence loss.

    The observation label itself is never changed.  We only spend more gradient on
    root pairs for which confusing/separating the pair changes the recovery decision:
    both roots have individually plausible recovery, but their best/shared recovery
    support differs.  This is the exact place where an observation-kernel error can
    create either an oracle-style false-safe (aliased incompatible roots separated)
    or a false-veto (distinguishable roots with different recovery choices merged).

    No regime identifier, bucket-specific threshold, or policy route is consumed.
    The returned tensor has shape ``[B,K,K]`` and is detached by construction.
    """
    if teacher_margins.ndim != 3:
        raise ValueError("teacher_margins must have shape [B,K,L]")
    b, k, _ = teacher_margins.shape
    rv = root_valid.bool()
    ov = option_valid.bool()
    if rv.shape != (b, k):
        raise ValueError("root_valid shape mismatch")
    tau = max(float(temperature), 1.0e-4)
    with torch.no_grad():
        finite = torch.isfinite(teacher_margins)
        valid = rv.unsqueeze(-1) & ov.unsqueeze(1) & finite
        margins = torch.nan_to_num(
            teacher_margins.detach().float(), nan=-20.0, posinf=20.0, neginf=-20.0
        )
        option_support = torch.sigmoid((margins - float(gamma)) / tau)
        option_support = torch.where(valid, option_support, torch.zeros_like(option_support))
        individual = option_support.amax(dim=-1)  # [B,K]
        # Soft mass that a *single* option is simultaneously viable for both roots.
        shared = (
            option_support.unsqueeze(2) * option_support.unsqueeze(1)
        ).amax(dim=-1)  # [B,K,K]
        independent = individual.unsqueeze(2) * individual.unsqueeze(1)
        conflict = (independent - shared).clamp(0.0, 1.0)
        pair_valid = rv.unsqueeze(2) & rv.unsqueeze(1)
        weights = 1.0 + max(0.0, float(conflict_scale)) * conflict
        weights = weights.clamp(1.0, max(1.0, float(max_weight)))
        weights = torch.where(pair_valid, weights, torch.ones_like(weights))
    return weights


def observation_consistent_frontier_calibration_loss(
    pred_r_dep: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    *,
    gamma: float = 0.0,
    option_temperature: float = 0.35,
    deployability_tolerance: float = 0.05,
    drs_tolerance: float = 0.05,
    sign_temperature: float = 0.08,
    regression_weight: float = 1.0,
    sign_weight: float = 0.50,
) -> torch.Tensor:
    """Calibrate the paper-native candidate-relative recovery frontier directly.

    Dense point-wise margin/Q losses can improve validation reconstruction while the
    deployment decision remains wrong because the selector is driven by *relative*
    non-compensatory coordinates.  This loss differentiates through OC-MERO's
    observation-conditioned option table and trains only two structural coordinates:

    1. nominal-minus-candidate deployability degradation in the same sigmoid space
       used by the component-veto teacher; and
    2. nominal-minus-candidate deployable-recovery-success (DRS) degradation, where
       DRS is the root-mass probability that each observation class has an admissible
       recovery option.

    The target and prediction use identical tolerances and grouping.  It is symmetric
    on both sides of zero (not a one-sided safe-positive or harmful-tail patch), and
    it never consumes a Safe/Near/Contact label.  This makes it different from the
    rejected v48.38 one-sided tail penalties and from generic candidate ranking.
    """
    if pred_q.ndim != 3 or teacher_q.ndim != 3:
        return pred_r_dep.sum() * 0.0
    b, k, _ = pred_q.shape
    if pred_r_dep.reshape(-1).numel() != b:
        raise ValueError("pred_r_dep batch mismatch")
    rv = root_valid.bool()
    ov = option_valid.bool()
    mask = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_r_dep.sum() * 0.0

    root_w = _root_weights(root_probs, root_valid)
    tau = max(float(option_temperature), 1.0e-4)
    pred_prob = torch.sigmoid(
        (torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau
    )
    pred_prob = torch.where(mask, pred_prob, torch.zeros_like(pred_prob))
    pred_exist = pred_prob.amax(dim=-1)
    pred_drs = (pred_exist * root_w).sum(dim=-1)
    teacher_exist = (((teacher_q.detach() >= float(gamma)) & mask).any(dim=-1)).float()
    teacher_drs = (teacher_exist * root_w.detach()).sum(dim=-1)

    prd = pred_r_dep.float().reshape(-1)
    trd = teacher_r_dep.detach().float().reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    nominal = is_nominal.reshape(-1) > 0.5
    finite = torch.isfinite(prd) & torch.isfinite(trd) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    pred_terms: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=-1)
    for key in torch.unique(keys[finite], dim=0):
        idx = torch.where(finite & (sh == key[0]) & (ti == key[1]))[0]
        noms = idx[nominal[idx]]
        recs = idx[~nominal[idx]]
        if noms.numel() != 1 or recs.numel() == 0:
            continue
        nom = noms[0]
        pred_dep_margin = (
            torch.sigmoid(prd[nom]).expand_as(prd[recs])
            - torch.sigmoid(prd[recs])
            - float(deployability_tolerance)
        )
        teacher_dep_margin = (
            torch.sigmoid(trd[nom]).expand_as(trd[recs])
            - torch.sigmoid(trd[recs])
            - float(deployability_tolerance)
        )
        pred_drs_margin = pred_drs[nom].expand_as(pred_drs[recs]) - pred_drs[recs] - float(drs_tolerance)
        teacher_drs_margin = teacher_drs[nom].expand_as(teacher_drs[recs]) - teacher_drs[recs] - float(drs_tolerance)
        pred_terms.append(torch.stack([pred_drs_margin, pred_dep_margin], dim=-1))
        target_terms.append(torch.stack([teacher_drs_margin, teacher_dep_margin], dim=-1))

    if not pred_terms:
        return pred_r_dep.sum() * 0.0
    pred = torch.cat(pred_terms, dim=0)
    target = torch.cat(target_terms, dim=0).detach()
    regression = F.smooth_l1_loss(pred, target)

    sign_tau = max(float(sign_temperature), 1.0e-4)
    sign_losses: list[torch.Tensor] = []
    for j in range(pred.shape[-1]):
        tgt = (target[:, j] > 0.0).float()
        logits = pred[:, j] / sign_tau
        pos = tgt > 0.5
        neg = ~pos
        if bool(pos.any()) and bool(neg.any()):
            weights = torch.where(
                pos,
                0.5 / pos.float().mean().clamp_min(1.0e-6),
                0.5 / neg.float().mean().clamp_min(1.0e-6),
            )
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt, weight=weights))
        else:
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt))
    sign_loss = torch.stack(sign_losses).mean()
    return float(regression_weight) * regression + float(sign_weight) * sign_loss


def decision_equivalent_frontier_calibration_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    pred_root_probs: torch.Tensor,
    teacher_root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    *,
    gamma: float = 0.0,
    option_temperature: float = 0.35,
    deployability_tolerance: float = 0.05,
    drs_tolerance: float = 0.05,
    gap_tolerance: float = 0.05,
    positive_gain: float = 0.015,
    sign_temperature: float = 0.08,
    regression_weight: float = 1.0,
    sign_weight: float = 0.50,
    pcd_weight: float = 1.0,
) -> torch.Tensor:
    """Forward-exact, backward-smooth calibration of deployed OC-MERO coordinates.

    v48.49 exposed a semantic gap: the deployment path uses a *hard* predicted DRS
    and exact PCD ``DRS * sigmoid(R_dep) * exp(-gap)``, while the DRFC witness and
    NAP transport used a smooth boundary mass.  Sharing a zero crossing is not
    sufficient when the downstream decision depends on coordinate magnitude and
    candidate-relative ordering.

    This loss therefore makes the forward values decision-equivalent to deployment:

    * predicted DRS uses the hard ``q_best >= gamma`` event and the model's own
      predicted root probabilities (the same weights used at inference);
    * deployability uses ``sigmoid(R_dep)``;
    * gap uses ``exp(-max(gap, 0))``; and
    * positive recovery value uses their exact multiplicative PCD.

    Gradients through the hard DRS event use a straight-through sigmoid surrogate,
    so no extra learned head, regime label, policy route, or threshold is introduced.
    The objective is symmetric on both sides of every decision boundary and only
    updates the existing recovery witness when used by the v48.50 witness stage.
    """
    if pred_q.ndim != 3 or teacher_q.ndim != 3:
        return pred_r_dep.sum() * 0.0
    b, k, _ = pred_q.shape
    rv = root_valid.bool()
    ov = option_valid.bool()
    if rv.shape != (b, k):
        raise ValueError("root_valid shape mismatch")
    pred_mask = rv.unsqueeze(-1) & ov.unsqueeze(1)
    teacher_mask = pred_mask & torch.isfinite(teacher_q)
    if not bool(teacher_mask.any()):
        return pred_r_dep.sum() * 0.0

    pred_w = _root_weights(pred_root_probs, root_valid)
    teacher_w = _root_weights(teacher_root_probs, root_valid).detach()
    tau = max(float(option_temperature), 1.0e-4)
    q_safe = torch.nan_to_num(pred_q.float(), nan=-20.0, posinf=20.0, neginf=-20.0)
    # Deployment validity is determined by root/option masks, not by whether a
    # teacher witness is finite. Teacher finiteness is used only on the target
    # side; otherwise training would silently give the predicted certificate a
    # teacher-dependent mask that inference never has.
    q_masked = torch.where(pred_mask, q_safe, q_safe.new_full((), -20.0))
    q_best = q_masked.amax(dim=-1)
    soft_exist = torch.sigmoid((q_best - float(gamma)) / tau)
    hard_exist = (q_best >= float(gamma)).to(dtype=soft_exist.dtype)
    # Forward hard, backward smooth.  This is a decision-coordinate surrogate,
    # not a replacement of the deployed coordinate.
    exist_st = hard_exist.detach() + soft_exist - soft_exist.detach()
    pred_drs = (pred_w * exist_st).sum(dim=-1).clamp(0.0, 1.0)

    teacher_exist = (((teacher_q.detach() >= float(gamma)) & teacher_mask).any(dim=-1)).float()
    teacher_drs = (teacher_w * teacher_exist).sum(dim=-1).clamp(0.0, 1.0)

    pred_dep = torch.sigmoid(pred_r_dep.float().reshape(-1))
    teacher_dep = torch.sigmoid(teacher_r_dep.detach().float().reshape(-1))
    pred_gap_quality = torch.exp(-torch.relu(pred_gap.float().reshape(-1)).clamp(max=20.0))
    teacher_gap = torch.relu(
        teacher_r_orc.detach().float().reshape(-1)
        - teacher_r_dep.detach().float().reshape(-1)
    )
    teacher_gap_quality = torch.exp(-teacher_gap.clamp(max=20.0))
    pred_pcd = (pred_drs * pred_dep * pred_gap_quality).clamp(0.0, 1.0)
    teacher_pcd = (teacher_drs * teacher_dep * teacher_gap_quality).clamp(0.0, 1.0).detach()

    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    nominal = is_nominal.reshape(-1) > 0.5
    finite = (
        torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
        & torch.isfinite(pred_dep) & torch.isfinite(teacher_dep)
        & torch.isfinite(pred_gap_quality) & torch.isfinite(teacher_gap_quality)
        & torch.isfinite(pred_pcd) & torch.isfinite(teacher_pcd)
    )
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    pred_terms: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=-1)
    for key in torch.unique(keys[finite], dim=0):
        idx = torch.where(finite & (sh == key[0]) & (ti == key[1]))[0]
        noms = idx[nominal[idx]]
        recs = idx[~nominal[idx]]
        if noms.numel() != 1 or recs.numel() == 0:
            continue
        nom = noms[0]
        pred_harm = torch.stack(
            [
                pred_drs[nom].expand_as(pred_drs[recs]) - pred_drs[recs] - float(drs_tolerance),
                pred_dep[nom].expand_as(pred_dep[recs]) - pred_dep[recs] - float(deployability_tolerance),
                pred_gap_quality[nom].expand_as(pred_gap_quality[recs]) - pred_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        teacher_harm = torch.stack(
            [
                teacher_drs[nom].expand_as(teacher_drs[recs]) - teacher_drs[recs] - float(drs_tolerance),
                teacher_dep[nom].expand_as(teacher_dep[recs]) - teacher_dep[recs] - float(deployability_tolerance),
                teacher_gap_quality[nom].expand_as(teacher_gap_quality[recs]) - teacher_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        pred_benefit = pred_pcd[recs] - pred_pcd[nom] - float(positive_gain)
        teacher_benefit = teacher_pcd[recs] - teacher_pcd[nom] - float(positive_gain)
        pred_terms.append(torch.cat([pred_harm, (float(pcd_weight) * pred_benefit).unsqueeze(-1)], dim=-1))
        target_terms.append(torch.cat([teacher_harm, (float(pcd_weight) * teacher_benefit).unsqueeze(-1)], dim=-1))

    if not pred_terms:
        return pred_r_dep.sum() * 0.0
    pred = torch.cat(pred_terms, dim=0)
    target = torch.cat(target_terms, dim=0).detach()
    regression = F.smooth_l1_loss(pred, target)

    sign_tau = max(float(sign_temperature), 1.0e-4)
    sign_losses: list[torch.Tensor] = []
    for j in range(pred.shape[-1]):
        tgt = (target[:, j] > 0.0).float()
        logits = pred[:, j] / sign_tau
        pos = tgt > 0.5
        neg = ~pos
        if bool(pos.any()) and bool(neg.any()):
            weights = torch.where(
                pos,
                0.5 / pos.float().mean().clamp_min(1.0e-6),
                0.5 / neg.float().mean().clamp_min(1.0e-6),
            )
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt, weight=weights))
        else:
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt))
    sign_loss = torch.stack(sign_losses).mean()
    return float(regression_weight) * regression + float(sign_weight) * sign_loss




def _physical_student_observation_consistent_success_st(
    pred_q: torch.Tensor,
    pred_margins: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Differentiable student analogue of the physical teacher DRS.

    The executed observation-consistent option is selected by the student's
    robust OC-MERO q row.  Root success is then evaluated on the student's
    *physical margin* for that selected option, matching the teacher/evaluator
    composition ``q selects -> margin sign certifies -> root mass aggregates``.

    Forward values are hard at the physical margin zero boundary.  Backward
    gradients use only a sigmoid STE on the selected margin; option selection
    remains the exact hard q selection, so no new soft router is introduced.
    """
    if pred_q.ndim != 3 or pred_margins.ndim != 3:
        return pred_q.reshape(pred_q.shape[0], -1).mean(dim=-1) * 0.0
    q = torch.nan_to_num(pred_q.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    m = torch.nan_to_num(pred_margins.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    b, k, l = q.shape
    if m.shape[:2] != (b, k):
        raise ValueError(f"pred_margins shape {tuple(m.shape)} incompatible with pred_q {tuple(q.shape)}")
    ll = min(l, m.shape[2])
    q = q[:, :, :ll]
    m = m[:, :, :ll]
    l = ll
    rv = root_valid.bool()[:, :k]
    ov = option_valid.bool()[:, :l]
    w = _root_weights(root_probs[:, :k], rv)
    valid = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(q)
    score = q.clamp(-5.0, 5.0) + 0.01 * ((q >= float(gamma)) & valid).float()
    score = torch.where(valid, score, torch.full_like(score, -1.0e9))
    opt = score.argmax(dim=-1)
    selected_margin = torch.gather(m, dim=2, index=opt.unsqueeze(-1)).squeeze(-1)
    selected_valid = rv & torch.gather(
        ov.unsqueeze(1).expand(-1, k, -1), 2, opt.unsqueeze(-1)
    ).squeeze(-1)
    tau = max(float(temperature), 1.0e-4)
    # Physical success is a margin-zero event. ``gamma`` belongs to the q-side
    # option-selection semantics only; moving the margin crossing with gamma
    # would break structural equivalence with both the teacher and deployment.
    soft = torch.sigmoid(selected_margin / tau)
    hard = (selected_margin >= 0.0).to(dtype=soft.dtype)
    success_st = hard.detach() + soft - soft.detach()
    success_st = torch.where(selected_valid, success_st, torch.zeros_like(success_st))
    return (w * success_st).sum(dim=-1).clamp(0.0, 1.0)


def selected_option_physical_boundary_distillation_loss(
    pred_margins: torch.Tensor,
    teacher_q: torch.Tensor,
    teacher_m_star: torch.Tensor,
    teacher_root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.08,
) -> torch.Tensor:
    """Distill the *selected* physical zero crossing without changing deployment.

    v48.53 showed that replacing the deployed q-hard DRS by a selected-margin
    physical DRS improves some candidate-level ranking/sensitivity diagnostics,
    but catastrophically increases harmful false-safe mass.  The useful signal
    is therefore treated as privileged training supervision rather than as a new
    hard certificate coordinate.

    The teacher OC-MERO q row selects the observation-consistent recovery option
    for each valid root/class.  Only that option's physical ``m_star == 0``
    boundary is distilled into the corresponding predicted margin.  Deployment
    remains byte-semantically q-hard; this loss never changes native DRS,
    admission thresholds, root logits, option routing, or regime behavior.

    The loss is root-probability weighted and class-balanced by *probability
    mass* rather than raw root count.  This keeps rare physical-positive and
    physical-negative boundaries from being erased while introducing no new
    tuned threshold: zero is the physical margin boundary, and ``temperature``
    reuses the existing frontier sign temperature.
    """
    if pred_margins.ndim != 3 or teacher_q.ndim != 3 or teacher_m_star.ndim != 3:
        return pred_margins.reshape(pred_margins.shape[0], -1).mean(dim=-1).mean() * 0.0
    b, k, l = teacher_q.shape
    if pred_margins.shape[:2] != (b, k) or teacher_m_star.shape[:2] != (b, k):
        raise ValueError(
            "selected-option physical boundary distillation shape mismatch: "
            f"pred_margins={tuple(pred_margins.shape)} teacher_q={tuple(teacher_q.shape)} "
            f"teacher_m_star={tuple(teacher_m_star.shape)}"
        )
    ll = min(l, pred_margins.shape[2], teacher_m_star.shape[2], option_valid.shape[-1])
    tq = torch.nan_to_num(teacher_q.detach().float()[:, :, :ll], nan=-1.0e9, posinf=5.0, neginf=-5.0)
    tm = torch.nan_to_num(teacher_m_star.detach().float()[:, :, :ll], nan=-1.0e9, posinf=5.0, neginf=-5.0)
    pm = torch.nan_to_num(pred_margins.float()[:, :, :ll], nan=-20.0, posinf=20.0, neginf=-20.0)
    rv = root_valid.bool()[:, :k]
    ov = option_valid.bool()[:, :ll]
    valid = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(teacher_q.detach()[:, :, :ll])
    if not bool(valid.any()):
        return pred_margins.sum() * 0.0

    # Keep the same deterministic teacher option-selection tie semantics used by
    # the observation-consistent q witness.  The physical margin is *not* used
    # to choose the option, so privileged future geometry cannot become a router.
    score = tq.clamp(-5.0, 5.0) + 0.01 * ((tq >= float(gamma)) & valid).float()
    score = torch.where(valid, score, torch.full_like(score, -1.0e9))
    opt = score.argmax(dim=-1)
    gather = opt.unsqueeze(-1)
    pred_selected = torch.gather(pm, 2, gather).squeeze(-1)
    teacher_selected = torch.gather(tm, 2, gather).squeeze(-1)
    teacher_margin_finite = torch.gather(
        torch.isfinite(teacher_m_star.detach()[:, :, :ll]), 2, gather
    ).squeeze(-1)
    selected_valid = rv & torch.gather(
        ov.unsqueeze(1).expand(-1, k, -1), 2, gather
    ).squeeze(-1) & teacher_margin_finite
    if not bool(selected_valid.any()):
        return pred_margins.sum() * 0.0

    target = (teacher_selected >= 0.0).float()
    tau = max(float(temperature), 1.0e-4)
    logits = pred_selected / tau
    root_w = _root_weights(teacher_root_probs[:, :k], rv).detach()
    root_w = root_w * selected_valid.float()

    # Balance by teacher root probability mass, not by root count.  This is
    # deterministic and regime-agnostic; when a minibatch contains one class
    # only, fall back to the unbalanced probability-weighted BCE.
    pos_mass = (root_w * target).sum()
    neg_mass = (root_w * (1.0 - target)).sum()
    if float(pos_mass.detach().cpu()) > 0.0 and float(neg_mass.detach().cpu()) > 0.0:
        class_w = torch.where(
            target > 0.5,
            0.5 / pos_mass.clamp_min(1.0e-8),
            0.5 / neg_mass.clamp_min(1.0e-8),
        )
        weights = root_w * class_w
        # The class-balanced weights sum to one by construction.
        return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum()
    denom = root_w.sum().clamp_min(1.0e-8)
    return (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * root_w
    ).sum() / denom

def boundary_complete_frontier_calibration_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    pred_root_probs: torch.Tensor,
    teacher_root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    *,
    gamma: float = 0.0,
    option_temperature: float = 0.35,
    deployability_tolerance: float = 0.05,
    drs_tolerance: float = 0.05,
    gap_tolerance: float = 0.05,
    positive_gain: float = 0.015,
    sign_temperature: float = 0.08,
    regression_weight: float = 1.0,
    sign_weight: float = 0.50,
    pcd_weight: float = 1.0,
    teacher_m_star: torch.Tensor | None = None,
    physical_teacher_sign_alignment: bool = False,
    pred_margins: torch.Tensor | None = None,
    physical_student_sign_alignment: bool = False,
    option_execution_semantics: str = "observation_class",
) -> torch.Tensor:
    """Boundary-complete calibration for one unified recovery certificate.

    v48.50 showed two complementary failure modes: a fully smooth frontier
    preserves local ranking but can miss the deployed material sign, whereas
    regressing the discontinuous hard DRS magnitude improves some signs but can
    destroy boundary-local ordering and veto true safe-positive recoveries.

    BC-DE therefore assigns the two coordinates different jobs without adding a
    head, regime input, selector, or tunable threshold:

    * **sign channel**: hard deployed DRS (forward hard / backward smooth STE),
      exact deployability, exact gap quality, and exact PCD supervise which side
      of each physical decision boundary the candidate belongs to;
    * **order channel**: boundary-resolved DRS and its smooth PCD supervise
      continuous magnitudes/ranking inside the hard equivalence classes.

    The existing ``positive_gain`` is reused as the global material-benefit
    boundary; all regimes share exactly the same primitive and parameters.

    With ``physical_teacher_sign_alignment=True`` (v48.52 PSA), only the
    *teacher sign* coordinate changes: q still selects the observation-consistent
    recovery action, while physical ``m_star >= 0`` determines root success.

    v48.53 adds the complementary student-side factor.  With
    ``physical_student_sign_alignment=True``, the student hard DRS uses the same
    composition: predicted q selects the legal option, the selected predicted
    physical margin owns the zero crossing, and predicted root mass aggregates
    success.  The smooth q-based order channel remains intentionally unchanged,
    preserving the v48.51-supported local ordering mechanism.
    """
    if pred_q.ndim != 3 or teacher_q.ndim != 3:
        return pred_r_dep.sum() * 0.0
    b, k, _ = pred_q.shape
    rv = root_valid.bool()
    ov = option_valid.bool()
    if rv.shape != (b, k):
        raise ValueError("root_valid shape mismatch")
    pred_mask = rv.unsqueeze(-1) & ov.unsqueeze(1)
    teacher_mask = pred_mask & torch.isfinite(teacher_q)
    if not bool(teacher_mask.any()):
        return pred_r_dep.sum() * 0.0

    pred_w = _root_weights(pred_root_probs, root_valid)
    teacher_w = _root_weights(teacher_root_probs, root_valid).detach()
    tau = max(float(option_temperature), 1.0e-4)

    pred_q_safe = torch.nan_to_num(pred_q.float(), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_q_best = torch.where(
        pred_mask, pred_q_safe, pred_q_safe.new_full((), -20.0)
    ).amax(dim=-1)
    pred_soft_exist = torch.sigmoid((pred_q_best - float(gamma)) / tau)
    pred_smooth_drs = (pred_w * pred_soft_exist).sum(dim=-1).clamp(0.0, 1.0)
    if bool(physical_student_sign_alignment):
        if pred_margins is None:
            raise ValueError(
                "physical_student_sign_alignment=true requires pred_margins"
            )
        if str(option_execution_semantics).strip().lower() not in {
            "observation_class", "observation_consistent", "ocmero"
        }:
            raise ValueError(
                "physical_student_sign_alignment currently requires observation_class semantics"
            )
        pred_hard_drs = _physical_student_observation_consistent_success_st(
            pred_q, pred_margins, pred_root_probs, root_valid, option_valid,
            gamma=float(gamma), temperature=float(option_temperature),
        )
    else:
        pred_hard_exist = (pred_q_best >= float(gamma)).to(dtype=pred_soft_exist.dtype)
        pred_exist_st = (
            pred_hard_exist.detach() + pred_soft_exist - pred_soft_exist.detach()
        )
        pred_hard_drs = (pred_w * pred_exist_st).sum(dim=-1).clamp(0.0, 1.0)

    teacher_q_safe = torch.nan_to_num(
        teacher_q.detach().float(), nan=-20.0, posinf=20.0, neginf=-20.0
    )
    teacher_q_best = torch.where(
        teacher_mask, teacher_q_safe, teacher_q_safe.new_full((), -20.0)
    ).amax(dim=-1)
    teacher_soft_exist = torch.sigmoid((teacher_q_best - float(gamma)) / tau)
    teacher_smooth_drs = (teacher_w * teacher_soft_exist).sum(dim=-1).clamp(0.0, 1.0)

    # v48.52/v48.53 sign teacher: teacher q selects the legal
    # observation-consistent recovery option; when PSA is enabled, success is
    # evaluated on the corresponding physical m_star margin.  v48.53 can also
    # align the student hard certificate to the same q-select -> margin-sign ->
    # root-mass composition above.  In either case, smooth q geometry remains
    # the order/magnitude channel.  No new head, threshold, regime input, or
    # policy router is introduced.
    if bool(physical_teacher_sign_alignment):
        if teacher_m_star is None:
            raise ValueError(
                "physical_teacher_sign_alignment=true requires teacher_m_star"
            )
        teacher_hard_drs = _exact_teacher_recovery_success(
            teacher_q.detach(),
            teacher_m_star.detach(),
            teacher_root_probs.detach(),
            root_valid,
            option_valid,
            gamma=float(gamma),
            semantics=str(option_execution_semantics),
        )
    else:
        teacher_hard_exist = (teacher_q_best >= float(gamma)).float()
        teacher_hard_drs = (teacher_w * teacher_hard_exist).sum(dim=-1).clamp(0.0, 1.0)

    pred_dep = torch.sigmoid(pred_r_dep.float().reshape(-1))
    teacher_dep = torch.sigmoid(teacher_r_dep.detach().float().reshape(-1))
    pred_gap_quality = torch.exp(-torch.relu(pred_gap.float().reshape(-1)).clamp(max=20.0))
    teacher_gap = torch.relu(
        teacher_r_orc.detach().float().reshape(-1)
        - teacher_r_dep.detach().float().reshape(-1)
    )
    teacher_gap_quality = torch.exp(-teacher_gap.clamp(max=20.0))

    pred_exact_pcd = (pred_hard_drs * pred_dep * pred_gap_quality).clamp(0.0, 1.0)
    teacher_exact_pcd = (
        teacher_hard_drs * teacher_dep * teacher_gap_quality
    ).clamp(0.0, 1.0).detach()
    pred_smooth_pcd = (pred_smooth_drs * pred_dep * pred_gap_quality).clamp(0.0, 1.0)
    teacher_smooth_pcd = (
        teacher_smooth_drs * teacher_dep * teacher_gap_quality
    ).clamp(0.0, 1.0).detach()

    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    nominal = is_nominal.reshape(-1) > 0.5
    finite = (
        torch.isfinite(pred_hard_drs) & torch.isfinite(teacher_hard_drs)
        & torch.isfinite(pred_smooth_drs) & torch.isfinite(teacher_smooth_drs)
        & torch.isfinite(pred_dep) & torch.isfinite(teacher_dep)
        & torch.isfinite(pred_gap_quality) & torch.isfinite(teacher_gap_quality)
        & torch.isfinite(pred_exact_pcd) & torch.isfinite(teacher_exact_pcd)
        & torch.isfinite(pred_smooth_pcd) & torch.isfinite(teacher_smooth_pcd)
    )
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    pred_order_terms: list[torch.Tensor] = []
    target_order_terms: list[torch.Tensor] = []
    pred_sign_terms: list[torch.Tensor] = []
    target_sign_terms: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=-1)
    for key in torch.unique(keys[finite], dim=0):
        idx = torch.where(finite & (sh == key[0]) & (ti == key[1]))[0]
        noms = idx[nominal[idx]]
        recs = idx[~nominal[idx]]
        if noms.numel() != 1 or recs.numel() == 0:
            continue
        nom = noms[0]

        # Continuous/order channel: only the DRS/PCD coordinates use the smooth
        # boundary resolution. DEP and gap are already continuous exact physical
        # coordinates and therefore appear unchanged in both channels.
        pred_order_harm = torch.stack(
            [
                pred_smooth_drs[nom].expand_as(pred_smooth_drs[recs]) - pred_smooth_drs[recs] - float(drs_tolerance),
                pred_dep[nom].expand_as(pred_dep[recs]) - pred_dep[recs] - float(deployability_tolerance),
                pred_gap_quality[nom].expand_as(pred_gap_quality[recs]) - pred_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        teacher_order_harm = torch.stack(
            [
                teacher_smooth_drs[nom].expand_as(teacher_smooth_drs[recs]) - teacher_smooth_drs[recs] - float(drs_tolerance),
                teacher_dep[nom].expand_as(teacher_dep[recs]) - teacher_dep[recs] - float(deployability_tolerance),
                teacher_gap_quality[nom].expand_as(teacher_gap_quality[recs]) - teacher_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        pred_order_benefit = pred_smooth_pcd[recs] - pred_smooth_pcd[nom] - float(positive_gain)
        teacher_order_benefit = teacher_smooth_pcd[recs] - teacher_smooth_pcd[nom] - float(positive_gain)
        pred_order_terms.append(
            torch.cat([pred_order_harm, (float(pcd_weight) * pred_order_benefit).unsqueeze(-1)], dim=-1)
        )
        target_order_terms.append(
            torch.cat([teacher_order_harm, (float(pcd_weight) * teacher_order_benefit).unsqueeze(-1)], dim=-1)
        )

        # Decision/sign channel: exact deployed hard DRS and exact PCD own the
        # zero crossing. This channel is never used for magnitude regression.
        pred_sign_harm = torch.stack(
            [
                pred_hard_drs[nom].expand_as(pred_hard_drs[recs]) - pred_hard_drs[recs] - float(drs_tolerance),
                pred_dep[nom].expand_as(pred_dep[recs]) - pred_dep[recs] - float(deployability_tolerance),
                pred_gap_quality[nom].expand_as(pred_gap_quality[recs]) - pred_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        teacher_sign_harm = torch.stack(
            [
                teacher_hard_drs[nom].expand_as(teacher_hard_drs[recs]) - teacher_hard_drs[recs] - float(drs_tolerance),
                teacher_dep[nom].expand_as(teacher_dep[recs]) - teacher_dep[recs] - float(deployability_tolerance),
                teacher_gap_quality[nom].expand_as(teacher_gap_quality[recs]) - teacher_gap_quality[recs] - float(gap_tolerance),
            ],
            dim=-1,
        )
        pred_sign_benefit = pred_exact_pcd[recs] - pred_exact_pcd[nom] - float(positive_gain)
        teacher_sign_benefit = teacher_exact_pcd[recs] - teacher_exact_pcd[nom] - float(positive_gain)
        pred_sign_terms.append(
            torch.cat([pred_sign_harm, (float(pcd_weight) * pred_sign_benefit).unsqueeze(-1)], dim=-1)
        )
        target_sign_terms.append(
            torch.cat([teacher_sign_harm, (float(pcd_weight) * teacher_sign_benefit).unsqueeze(-1)], dim=-1)
        )

    if not pred_order_terms:
        return pred_r_dep.sum() * 0.0
    pred_order = torch.cat(pred_order_terms, dim=0)
    target_order = torch.cat(target_order_terms, dim=0).detach()
    pred_sign = torch.cat(pred_sign_terms, dim=0)
    target_sign = torch.cat(target_sign_terms, dim=0).detach()
    regression = F.smooth_l1_loss(pred_order, target_order)

    sign_tau = max(float(sign_temperature), 1.0e-4)
    sign_losses: list[torch.Tensor] = []
    for j in range(pred_sign.shape[-1]):
        tgt = (target_sign[:, j] > 0.0).float()
        logits = pred_sign[:, j] / sign_tau
        pos = tgt > 0.5
        neg = ~pos
        if bool(pos.any()) and bool(neg.any()):
            weights = torch.where(
                pos,
                0.5 / pos.float().mean().clamp_min(1.0e-6),
                0.5 / neg.float().mean().clamp_min(1.0e-6),
            )
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt, weight=weights))
        else:
            sign_losses.append(F.binary_cross_entropy_with_logits(logits, tgt))
    sign_loss = torch.stack(sign_losses).mean()
    return float(regression_weight) * regression + float(sign_weight) * sign_loss

def observation_class_option_success_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Calibrate existential recovery success for each observation class.

    Each row of OC-MERO q is already an observation-conditioned lower-tail
    witness over compatible roots.  The target here is therefore whether that
    class has *some* valid option, rather than whether one option works across
    all distinguishable classes.
    """
    if pred_q.ndim != 3 or teacher_q.ndim != 3:
        return pred_q.sum() * 0.0
    mask = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    tau = max(float(temperature), 1.0e-3)
    pred_prob = torch.sigmoid((torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) - float(gamma)) / tau)
    pred_prob = torch.where(mask, pred_prob, torch.zeros_like(pred_prob))
    teacher_success = ((teacher_q.detach() >= float(gamma)) & mask).any(dim=-1).float()
    pred_exist = pred_prob.max(dim=-1).values.clamp(1.0e-4, 1.0 - 1.0e-4)
    rv = root_valid.bool()
    root_w = _root_weights(root_probs, root_valid)
    valid = rv & mask.any(dim=-1)
    if not bool(valid.any()):
        return pred_q.sum() * 0.0
    loss = F.binary_cross_entropy(pred_exist, teacher_success, reduction="none")
    denom = (root_w * valid.float()).sum().clamp_min(1.0e-8)
    return (loss * root_w * valid.float()).sum() / denom


def observation_class_best_option_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    """Root/class-weighted CE for the OC-MERO recovery-option witness.

    Distinguishable post-prefix observations may choose different options; only
    observationally compatible roots are coupled inside each q row.  This is
    the option-selection semantics used by Eq. OC-MERO in the paper.
    """
    if pred_q.ndim != 3 or teacher_q.ndim != 3:
        return pred_q.sum() * 0.0
    b, k, l = pred_q.shape
    ov = option_valid.bool()
    mask = root_valid.bool().unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(teacher_q)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    teacher = torch.nan_to_num(teacher_q.detach(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    # A tiny gamma-success tie-break makes the target stable around zero while
    # preserving signed q as the primary ranking statistic.
    tscore = teacher.clamp(-5.0, 5.0) + 0.01 * ((teacher >= float(gamma)) & mask).float()
    tscore = torch.where(mask, tscore, torch.full_like(tscore, -1.0e9))
    target = tscore.argmax(dim=-1)
    tau = max(float(temperature), 1.0e-3)
    logits = torch.nan_to_num(pred_q, nan=-20.0, posinf=20.0, neginf=-20.0) / tau
    logits = torch.where(ov.unsqueeze(1), logits, torch.full_like(logits, -1.0e4))
    ce = F.cross_entropy(logits.reshape(b * k, l), target.reshape(b * k), reduction="none").reshape(b, k)
    valid = root_valid.bool() & mask.any(dim=-1)
    root_w = _root_weights(root_probs, root_valid)
    denom = (root_w * valid.float()).sum().clamp_min(1.0e-8)
    return (ce * root_w * valid.float()).sum() / denom



def groupwise_candidate_ranking_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_artifact: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    candidate_index: torch.Tensor,
    *,
    margin: float = 0.25,
    gap_weight: float = 0.25,
    teacher_gap_weight: float = 0.25,
    artifact_only: bool = True,
) -> torch.Tensor:
    """Rank candidates within the same scene-time group.

    The selector must choose a prefix in a scene-time candidate set, but the
    pointwise losses only teach absolute margins.  This loss pushes the learned
    score of the teacher-best deployable candidate above false-recoverable /
    oracle-artifact candidates from the same scene-time group.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = pred_r_dep.float().reshape(-1)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=5.0, neginf=0.0), min=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    ta = teacher_artifact.float().reshape(-1) > 0.5
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    ci = candidate_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), trd.numel(), tro.numel(), ta.numel(), sh.numel(), ti.numel(), ci.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, trd, tro, ta, sh, ti, ci = pr[:n], pg[:n], trd[:n], tro[:n], ta[:n], sh[:n], ti[:n], ci[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap
    pred_score = pr - float(gap_weight) * pg
    losses: list[torch.Tensor] = []
    # Batch sizes are small; explicit grouping keeps the logic readable and CPU/GPU safe.
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        # Prefer deployable teacher candidates; if none exists, rank by least-bad teacher score.
        deployable = idx[trd[idx] >= 0.0]
        pool = deployable if deployable.numel() > 0 else idx
        best_local = pool[torch.argmax(teacher_score[pool])]
        neg = idx[(tro[idx] >= 0.0) & (trd[idx] < 0.0)]
        if bool(artifact_only):
            neg = idx[ta[idx] | ((tro[idx] >= 0.0) & (trd[idx] < 0.0))]
        else:
            worse = teacher_score[idx] < teacher_score[best_local] - 1.0e-6
            neg = idx[worse | ta[idx] | ((tro[idx] >= 0.0) & (trd[idx] < 0.0))]
        neg = neg[neg != best_local]
        if neg.numel() == 0:
            continue
        diff = pred_score[best_local] - pred_score[neg]
        losses.append(F.relu(float(margin) - diff).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()

def groupwise_candidate_ce_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    *,
    temperature: float = 0.35,
    pred_gap_weight: float = 0.35,
    teacher_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    require_deployable_target: bool = True,
) -> torch.Tensor:
    """Cross-entropy over candidates in each scene-time group.

    Pairwise artifact ranking is not enough for contact recovery: the selected
    teacher action can be a non-artifact candidate whose advantage is higher
    post-contact deployability, not merely avoiding an oracle artifact.  This
    loss treats the scene-time candidate set as the decision space and directly
    teaches the learned candidate score to put its probability mass on the
    teacher-best observation-consistent deployable candidate.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    # Utility is deliberately weak here.  The teacher decision should primarily
    # reflect deployable recovery, but small utility ties keep benign scenes
    # nominal-preserving.
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    tau = max(float(temperature), 1.0e-3)
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        target_pool = idx[trd[idx] >= 0.0] if bool(require_deployable_target) else idx
        if target_pool.numel() == 0:
            target_pool = idx
        best = target_pool[torch.argmax(teacher_score[target_pool])]
        local_target = int((idx == best).nonzero(as_tuple=False).reshape(-1)[0].item())
        logits = pred_score[idx].reshape(1, -1) / tau
        target = torch.tensor([local_target], device=pred_r_dep.device, dtype=torch.long)
        losses.append(F.cross_entropy(logits, target))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()



def groupwise_score_distillation_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    *,
    pred_gap_weight: float = 0.45,
    teacher_gap_weight: float = 0.45,
    utility_weight: float = 0.02,
    teacher_temperature: float = 0.20,
    pred_temperature: float = 0.30,
    min_group_size: int = 2,
) -> torch.Tensor:
    """Dense distillation of the teacher candidate ranking within a scene-time.

    v12 used a hard one-hot target for the single teacher-best candidate.  That
    gives sparse gradients and ignores the useful ordering among non-artifact
    deployable alternatives.  This KL loss transfers the full teacher ranking
    distribution over the candidate group, making contact recovery less dependent
    on artifact-only negatives.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    tt = max(float(teacher_temperature), 1.0e-3)
    pt = max(float(pred_temperature), 1.0e-3)
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < int(min_group_size):
            continue
        idx = torch.where(mask)[0]
        target = torch.softmax(teacher_score[idx] / tt, dim=0).detach()
        logp = torch.log_softmax(pred_score[idx] / pt, dim=0)
        losses.append(F.kl_div(logp, target, reduction="batchmean"))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def _differentiable_shared_success(
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Differentiable proxy for globally shared option success per candidate."""
    if pred_q.ndim != 3:
        return pred_q.reshape(pred_q.shape[0], -1).mean(dim=-1) * 0.0
    w = torch.clamp(root_probs.float(), min=0.0) * root_valid.bool().float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    valid = option_valid.bool().unsqueeze(1)
    logits = (pred_q.float() - float(gamma)) / max(float(temperature), 1.0e-3)
    succ = torch.sigmoid(torch.nan_to_num(logits, nan=-20.0, posinf=20.0, neginf=-20.0))
    succ = torch.where(valid, succ, torch.zeros_like(succ))
    shared = (w.unsqueeze(-1) * succ).sum(dim=1)
    shared = torch.where(option_valid.bool(), shared, torch.full_like(shared, -1.0))
    return torch.clamp(shared.max(dim=-1).values, min=0.0, max=1.0)





def _differentiable_observation_consistent_success(
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Differentiable DRS proxy with one option per observation-conditioned row."""
    if pred_q.ndim != 3:
        return pred_q.reshape(pred_q.shape[0], -1).mean(dim=-1) * 0.0
    w = torch.clamp(root_probs.float(), min=0.0) * root_valid.bool().float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    valid = root_valid.bool().unsqueeze(-1) & option_valid.bool().unsqueeze(1)
    logits = (pred_q.float() - float(gamma)) / max(float(temperature), 1.0e-3)
    succ = torch.sigmoid(torch.nan_to_num(logits, nan=-20.0, posinf=20.0, neginf=-20.0))
    succ = torch.where(valid, succ, torch.full_like(succ, -1.0))
    per_class = succ.max(dim=-1).values.clamp(0.0, 1.0)
    return (w * per_class).sum(dim=-1).clamp(0.0, 1.0)


def _recovery_success_proxy(
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    temperature: float = 0.25,
    semantics: str = "global",
) -> torch.Tensor:
    mode = str(semantics).strip().lower()
    if mode in {"observation_class", "observation_consistent", "ocmero"}:
        return _differentiable_observation_consistent_success(
            pred_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=temperature
        )
    if mode in {"global", "global_shared", "single_global"}:
        return _differentiable_shared_success(
            pred_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=temperature
        )
    raise ValueError(f"Unknown option execution semantics: {semantics!r}")





def _exact_teacher_shared_success(
    teacher_q: torch.Tensor,
    teacher_m_star: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Exact hard shared-option DRS used by calibration, in batched Torch form.

    The selected option maximizes root-probability success mass under teacher-q,
    with the same 0.01 mean-value tie-break as ``best_shared_option_index``.
    Success is then evaluated on teacher ``m_star`` exactly as deployment
    calibration does.  The result is a detached supervision target.
    """
    if teacher_q.ndim != 3 or teacher_m_star.ndim != 3:
        return teacher_q.reshape(teacher_q.shape[0], -1).mean(dim=-1) * 0.0
    q = torch.nan_to_num(teacher_q.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    m = torch.nan_to_num(teacher_m_star.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    b, k, l = q.shape
    if m.shape[:2] != (b, k):
        raise ValueError(f"teacher_m_star shape {tuple(m.shape)} incompatible with teacher_q {tuple(q.shape)}")
    if m.shape[2] != l:
        ll = min(l, m.shape[2])
        q = q[:, :, :ll]
        m = m[:, :, :ll]
        l = ll
    w = torch.clamp(root_probs.float()[:, :k], min=0.0)
    rv = root_valid.bool()[:, :k]
    w = w * rv.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    ov = option_valid.bool()[:, :l]
    finite = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(q)
    success_mass = (((q >= float(gamma)) & finite).float() * w.unsqueeze(-1)).sum(dim=1)
    value_score = (torch.where(finite, q.clamp(-5.0, 5.0), torch.zeros_like(q)) * w.unsqueeze(-1)).sum(dim=1)
    option_score = success_mass + 0.01 * value_score
    option_score = torch.where(ov, option_score, torch.full_like(option_score, -1.0e9))
    opt = option_score.argmax(dim=-1)
    gather_idx = opt.view(b, 1, 1).expand(-1, k, 1)
    selected_margin = torch.gather(m, dim=2, index=gather_idx).squeeze(-1)
    selected_valid = rv & torch.gather(ov.unsqueeze(1).expand(-1, k, -1), 2, gather_idx).squeeze(-1)
    success = selected_valid & torch.isfinite(selected_margin) & (selected_margin >= 0.0)
    return (w * success.float()).sum(dim=-1).clamp(0.0, 1.0).detach()


def _exact_teacher_observation_consistent_success(
    teacher_q: torch.Tensor,
    teacher_m_star: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Hard teacher DRS with recovery selected by observation-conditioned q row."""
    if teacher_q.ndim != 3 or teacher_m_star.ndim != 3:
        return teacher_q.reshape(teacher_q.shape[0], -1).mean(dim=-1) * 0.0
    q = torch.nan_to_num(teacher_q.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    m = torch.nan_to_num(teacher_m_star.float(), nan=-1.0e9, posinf=5.0, neginf=-5.0)
    b, k, l = q.shape
    ll = min(l, m.shape[2])
    q = q[:, :, :ll]; m = m[:, :, :ll]; l = ll
    rv = root_valid.bool()[:, :k]
    ov = option_valid.bool()[:, :l]
    w = torch.clamp(root_probs.float()[:, :k], min=0.0) * rv.float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    valid = rv.unsqueeze(-1) & ov.unsqueeze(1) & torch.isfinite(q)
    score = q.clamp(-5.0, 5.0) + 0.01 * ((q >= float(gamma)) & valid).float()
    score = torch.where(valid, score, torch.full_like(score, -1.0e9))
    opt = score.argmax(dim=-1)
    selected_margin = torch.gather(m, dim=2, index=opt.unsqueeze(-1)).squeeze(-1)
    selected_valid = rv & torch.gather(ov.unsqueeze(1).expand(-1, k, -1), 2, opt.unsqueeze(-1)).squeeze(-1)
    success = selected_valid & torch.isfinite(selected_margin) & (selected_margin >= 0.0)
    return (w * success.float()).sum(dim=-1).clamp(0.0, 1.0).detach()


def _exact_teacher_recovery_success(
    teacher_q: torch.Tensor,
    teacher_m_star: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    *,
    gamma: float = 0.0,
    semantics: str = "global",
) -> torch.Tensor:
    mode = str(semantics).strip().lower()
    if mode in {"observation_class", "observation_consistent", "ocmero"}:
        return _exact_teacher_observation_consistent_success(
            teacher_q, teacher_m_star, root_probs, root_valid, option_valid, gamma=gamma
        )
    if mode in {"global", "global_shared", "single_global"}:
        return _exact_teacher_shared_success(
            teacher_q, teacher_m_star, root_probs, root_valid, option_valid, gamma=gamma
        )
    raise ValueError(f"Unknown option execution semantics: {semantics!r}")

def macro_shared_success_calibration_loss(
    pred_q: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    macro_type_id: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    gamma: float = 0.0,
    temperature: float = 0.25,
    pos_threshold: float = 0.80,
    neg_threshold: float = 0.05,
    pos_weight: float = 4.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """Macro-conditioned calibration for the shared recovery-option success head.

    v21 could rank a few protective macros, but the selector still missed true
    brake cases because their predicted shared-option success sometimes stayed
    near zero even when the teacher shared option succeeded.  This loss supervises
    the candidate-level DRS proxy used by the selector, only for semantic recovery
    macros in near/contact buckets.  Ambiguous middle-success cases are ignored so
    the gradients focus on high-confidence deployable and non-deployable macros.
    """
    if pred_q.ndim != 3 or pred_q.shape[0] == 0 or not macro_ids or not bucket_ids:
        return pred_q.reshape(pred_q.shape[0], -1).sum() * 0.0
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=gamma, temperature=max(0.08, temperature * 0.5)
        ).reshape(-1)
    mac = macro_type_id.reshape(-1)
    bid = bucket_id.reshape(-1)
    n = min(pred_drs.numel(), teacher_drs.numel(), mac.numel(), bid.numel())
    if n <= 0:
        return pred_q.sum() * 0.0
    pred_drs, teacher_drs, mac, bid = pred_drs[:n], teacher_drs[:n], mac[:n], bid[:n]
    finite = torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    high = teacher_drs >= float(pos_threshold)
    low = teacher_drs <= float(neg_threshold)
    mask = finite & macro_mask & bucket_mask & (high | low)
    if not bool(mask.any()):
        return pred_q.sum() * 0.0
    target = high.float()
    p = pred_drs.clamp(1.0e-4, 1.0 - 1.0e-4)
    weights = torch.where(high, torch.full_like(p, float(pos_weight)), torch.full_like(p, float(neg_weight)))
    loss = F.binary_cross_entropy(p[mask], target[mask], weight=weights[mask], reduction="mean")
    return loss

def _torch_pcd_score(drs: torch.Tensor, r_dep: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    """Torch analogue of evaluation.metrics.post_contact_deployability_score."""
    return torch.clamp(drs, 0.0, 1.0) * torch.sigmoid(r_dep.float()) * torch.exp(-torch.clamp(gap.float(), min=0.0, max=20.0))




def deployability_dominance_calibration_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (2,),
    margin: float = 0.12,
    min_teacher_pcd_gain: float = 0.04,
    min_teacher_best_pcd: float = 0.50,
    max_nominal_teacher_pcd: float = 0.62,
    pred_gap_weight: float = 0.20,
    pred_drs_weight: float = 0.35,
    utility_weight: float = 0.00,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    target_min_pred_pcd: float = 0.45,
    nominal_max_pred_pcd: float = 0.55,
) -> torch.Tensor:
    """Regime-conditioned deployability-dominance calibration.

    Selector-only rescue can only use learned deployability heads.  The v26
    audits show that those heads still rank a high-confidence nominal prefix
    above teacher-deployable recovery macros in some post-contact states, while
    in near-contact they over-admit brake when merge/yield/nominal is the true
    deployable choice.  This loss distills the *group-wise* teacher PCD ordering
    directly into the model: when a paper-eligible recovery macro has materially
    higher teacher PCD than nominal, its predicted PCD score must outrank nominal;
    when nominal teacher PCD is poor, the nominal predicted PCD is also capped.
    """
    if pred_r_dep.numel() <= 1 or not macro_ids or not bucket_ids:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5)
        ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro = pr[:n], pg[:n], u[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap)
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    # PCD is the primary target; the auxiliary score retains margin/gap/utility
    # signals so gradients do not collapse when PCD values tie near 0 or 0.62.
    pred_score = pred_pcd + pred_drs_weight * pred_drs + pr - pred_gap_weight * pg + utility_weight * u
    teacher_score = teacher_pcd + 0.25 * teacher_drs + trd - 0.10 * teacher_gap
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        rec_idx = idx[macro_mask[idx]]
        if rec_idx.numel() == 0:
            continue
        good = rec_idx[
            (teacher_pcd[rec_idx] >= float(min_teacher_best_pcd))
            & (teacher_pcd[rec_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        ]
        if good.numel() == 0:
            continue
        target = good[torch.argmax(teacher_score[good])]
        losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
        losses.append(F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
        losses.append(F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
        # Separate from other non-target candidates in the group, but with a
        # smaller margin so legitimate ties among recovery families are allowed.
        others = idx[idx != target]
        if others.numel() > 0:
            losses.append(F.relu(0.5 * float(margin) - (pred_score[target] - pred_score[others])).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def direct_teacher_pcd_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    positive_macro_ids: tuple[int, ...] | None = None,
    bucket_ids: tuple[int, ...] = (2,),
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    regression_weight: float = 1.0,
    ranking_weight: float = 2.5,
    nominal_penalty_weight: float = 1.0,
    false_positive_weight: float = 1.5,
    margin: float = 0.18,
    min_teacher_pcd_gain: float = 0.015,
    min_teacher_best_pcd: float = 0.50,
    max_nominal_teacher_pcd: float = 0.68,
    target_min_pred_pcd: float = 0.52,
    nominal_max_pred_pcd: float = 0.50,
    focus_non_nominal_weight: float = 2.0,
    false_positive_margin: float = 0.03,
    component_weight: float = 0.0,
    positive_component_weight: float = 0.0,
    nominal_cap_weight: float = 1.0,
    positive_rank_all_weight: float = 0.0,
    positive_floor_weight: float = 0.0,
    positive_min_pred_r_dep: float = -1.0e9,
    positive_max_pred_gap: float = -1.0,
    positive_min_pred_drs: float = -1.0,
) -> torch.Tensor:
    """Directly distill teacher post-contact deployability into learned PCD.

    v30 audits showed a selector-side failure that cannot be fixed by loosening
    thresholds alone: in contact groups, paper-best brake/yield candidates can
    have higher teacher PCD while the learned heads still rank nominal or an
    unrelated merge/stabilize candidate higher.  DDC only trains a sparse
    dominance event.  This loss supervises the *absolute* teacher PCD for all
    candidates in the requested regimes and adds group-wise ranking/anti-false-
    positive terms around nominal.

    Macro ids follow prefix_generation.MACROS:
    nominal=0, brake=2, yield=3, merge=5, stabilize=7.

    v32 adds a separate positive_macro_ids argument.  macro_ids still controls
    dense regression and anti-false-positive suppression, while positive_macro_ids
    controls which recovery families are allowed to be promoted above nominal.
    This lets us keep merge/stabilize calibrated low without letting them compete
    with the contact brake/yield frontier.
    """
    if pred_r_dep.numel() <= 1 or not bucket_ids:
        return pred_r_dep.sum() * 0.0

    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5)
        ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro = pr[:n], pg[:n], u[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]

    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == int(m)
    pos_ids = macro_ids if positive_macro_ids is None else positive_macro_ids
    positive_macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in pos_ids):
        positive_macro_mask |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap).detach()
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)

    # Absolute regression is what v30 was missing: bad merge/stabilize rescue
    # candidates should be explicitly regressed to low teacher PCD, not only
    # filtered at selector time.
    reg_mask = finite
    reg_w = torch.ones_like(pred_pcd)
    reg_w = torch.where(macro_mask & finite, reg_w * float(focus_non_nominal_weight), reg_w)
    reg = F.smooth_l1_loss(pred_pcd[reg_mask], teacher_pcd[reg_mask], reduction="none")
    reg_loss = (reg * reg_w[reg_mask]).sum() / reg_w[reg_mask].sum().clamp_min(1.0e-6)

    # Component-level supervision keeps gradients alive when the multiplicative
    # PCD proxy is wrong for several reasons at once (the v31 contact misses had
    # brake DRS high but learned R_dep too low and gap too high).
    comp_mask = finite & (macro_mask | isn)
    if bool(comp_mask.any()):
        comp = F.smooth_l1_loss(pr[comp_mask], trd[comp_mask], reduction="none")
        comp = comp + 0.5 * F.smooth_l1_loss(pg[comp_mask], teacher_gap[comp_mask], reduction="none")
        comp = comp + F.smooth_l1_loss(pred_drs[comp_mask], teacher_drs[comp_mask], reduction="none")
        comp_loss = comp.mean()
    else:
        comp_loss = pred_r_dep.sum() * 0.0

    rank_losses: list[torch.Tensor] = []
    nominal_losses: list[torch.Tensor] = []
    false_pos_losses: list[torch.Tensor] = []
    pred_score = pred_pcd + 0.20 * pred_drs + pr - 0.10 * pg + 0.01 * u
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        group = finite & (sh == key[0]) & (ti == key[1])
        if int(group.sum().item()) < 2:
            continue
        idx = torch.where(group)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        rec_idx = idx[macro_mask[idx]]
        pos_rec_idx = idx[positive_macro_mask[idx]]
        if rec_idx.numel() == 0 or pos_rec_idx.numel() == 0:
            continue

        good = pos_rec_idx[
            (teacher_pcd[pos_rec_idx] >= float(min_teacher_best_pcd))
            & (teacher_pcd[pos_rec_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        ]
        if good.numel() > 0:
            target = good[torch.argmax(teacher_pcd[good])]
            rank_losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
            rank_losses.append(F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
            if float(positive_rank_all_weight) > 0.0:
                rank_losses.append(float(positive_rank_all_weight) * F.relu(float(margin) - (pred_score[good] - pred_score[nom])).mean())
            if float(positive_floor_weight) > 0.0:
                floor_terms = [F.relu(float(target_min_pred_pcd) - pred_pcd[good])]
                if float(positive_min_pred_r_dep) > -1.0e8:
                    floor_terms.append(F.relu(float(positive_min_pred_r_dep) - pr[good]))
                if float(positive_max_pred_gap) >= 0.0:
                    floor_terms.append(F.relu(pg[good] - float(positive_max_pred_gap)))
                if float(positive_min_pred_drs) >= 0.0:
                    floor_terms.append(F.relu(float(positive_min_pred_drs) - pred_drs[good]))
                rank_losses.append(float(positive_floor_weight) * torch.stack([t.mean() for t in floor_terms]).mean())
            if float(positive_component_weight) > 0.0:
                rank_losses.append(float(positive_component_weight) * F.smooth_l1_loss(pr[target], trd[target]))
                rank_losses.append(float(positive_component_weight) * 0.5 * F.smooth_l1_loss(pg[target], teacher_gap[target]))
                rank_losses.append(float(positive_component_weight) * F.smooth_l1_loss(pred_drs[target], teacher_drs[target]))
            nominal_losses.append(float(nominal_cap_weight) * F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
            worse = idx[(idx != target) & (teacher_pcd[idx] <= teacher_pcd[target] - float(min_teacher_pcd_gain))]
            if worse.numel() > 0:
                rank_losses.append(F.relu(0.5 * float(margin) - (pred_score[target] - pred_score[worse])).mean())

        # Anti false-positive: if nominal is better than a recovery macro under
        # teacher PCD, do not let learned PCD fabricate a rescue.  This targets
        # the v30 bad merge/stabilize rows where paper-best was nominal.
        bad_rec = rec_idx[teacher_pcd[nom] >= teacher_pcd[rec_idx] + float(false_positive_margin)]
        if bad_rec.numel() > 0:
            false_pos_losses.append(F.relu(float(margin) - (pred_score[nom] - pred_score[bad_rec])).mean())

    zero = pred_r_dep.sum() * 0.0
    rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
    nominal_loss = torch.stack(nominal_losses).mean() if nominal_losses else zero
    false_pos_loss = torch.stack(false_pos_losses).mean() if false_pos_losses else zero
    return (
        float(regression_weight) * reg_loss
        + float(ranking_weight) * rank_loss
        + float(nominal_penalty_weight) * nominal_loss
        + float(false_positive_weight) * false_pos_loss
        + float(component_weight) * comp_loss
    )

def protective_macro_recovery_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 7),
    bucket_ids: tuple[int, ...] = (2,),
    margin: float = 0.14,
    min_teacher_r_dep: float = 0.0,
    min_teacher_drs: float = 0.50,
    min_teacher_pcd_gain: float = 0.02,
    max_nominal_teacher_pcd: float = 0.90,
    pred_gap_weight: float = 0.18,
    pred_drs_weight: float = 0.65,
    utility_weight: float = 0.02,
    teacher_gap_weight: float = 0.10,
    teacher_drs_weight: float = 0.70,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    target_min_pred_drs: float = 0.62,
) -> torch.Tensor:
    """Macro-conditioned supervision for low-headroom protective recovery.

    The selector in contact regimes is supposed to admit brake/stabilize only
    when they are semantically protective *and* deployably recoverable.  Pointwise
    R_dep regression under-trains this case because the useful brake/stabilize
    candidate is rare inside each scene-time candidate set.  This loss activates
    only when the teacher-labelled group contains a protective macro whose
    post-contact deployability (DRS * margin gate * oracle-gap discount) is better
    than nominal.  It then ranks that macro above nominal and non-protective
    alternatives using the same deployability coordinates used by the selector.
    """
    if pred_r_dep.numel() <= 1 or not macro_ids or not bucket_ids:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5)
        ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, pred_drs, teacher_drs = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == int(b)
    macro_mask_all = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask_all |= mac == int(m)
    finite = finite & bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap)
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(teacher_drs_weight) * teacher_drs + teacher_pcd
    pred_score = pr - float(pred_gap_weight) * pg + float(pred_drs_weight) * pred_drs + pred_pcd + float(utility_weight) * u
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_idx = idx[isn[idx]]
        if nom_idx.numel() == 0:
            continue
        nom = nom_idx[0]
        prot_idx = idx[macro_mask_all[idx]]
        if prot_idx.numel() == 0:
            continue
        teacher_ok = (
            (trd[prot_idx] >= float(min_teacher_r_dep))
            & (teacher_drs[prot_idx] >= float(min_teacher_drs))
            & (teacher_pcd[prot_idx] >= teacher_pcd[nom] + float(min_teacher_pcd_gain))
            & (teacher_pcd[nom] <= float(max_nominal_teacher_pcd))
        )
        if not bool(teacher_ok.any()):
            continue
        cand = prot_idx[teacher_ok]
        target = cand[torch.argmax(teacher_score[cand])]
        # Rank the teacher-certified protective macro above nominal.
        losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nom])))
        # Also separate it from non-protective alternatives in the same group,
        # but do not punish other protective candidates (they may be ties).
        nonprot = idx[(idx != target) & (~macro_mask_all[idx])]
        if nonprot.numel() > 0:
            losses.append(F.relu(float(margin) - (pred_score[target] - pred_score[nonprot])).mean())
        # The target should look deployable to the selector in both aggregate
        # margin and shared-option success coordinates.
        losses.append(F.relu(float(min_teacher_r_dep) - pr[target]))
        losses.append(F.relu(float(target_min_pred_drs) - pred_drs[target]))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def safe_nominal_preservation_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    pred_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    margin: float = 0.18,
    pred_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    drs_weight: float = 0.30,
    min_nominal_success: float = 0.90,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
) -> torch.Tensor:
    """Learn a no-recovery certificate on safe/background groups.

    This is stronger than the v12 hard nominal lock: on safe groups, the model is
    explicitly trained to score the nominal prefix above recovery prefixes and to
    keep the nominal shared-option success proxy high.  At inference, the
    selector can then abstain from intervention when no deployable recovery is
    admitted, instead of blindly falling back to a high-utility recovery action.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid, gamma=success_gamma, temperature=success_temperature
    ).reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), u.numel(), drs.numel(), sh.numel(), ti.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, drs, sh, ti, isn, bid = pr[:n], pg[:n], u[:n], drs[:n], sh[:n], ti[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(drs) & (bid == 0)
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u + float(drs_weight) * drs
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_local = idx[isn[idx]]
        if nom_local.numel() == 0:
            continue
        nom = nom_local[0]
        others = idx[idx != nom]
        if others.numel() > 0:
            losses.append(F.relu(float(margin) - (pred_score[nom] - pred_score[others])).mean())
        losses.append(F.relu(float(min_nominal_success) - drs[nom]))
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()


def nominal_switch_consistency_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    utility: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    is_nominal: torch.Tensor,
    *,
    margin: float = 0.12,
    pred_gap_weight: float = 0.30,
    teacher_gap_weight: float = 0.35,
    utility_weight: float = 0.03,
    teacher_gain_margin: float = 0.06,
    nominal_deployable_gamma: float = 0.0,
    nominal_gap_max: float = 0.30,
) -> torch.Tensor:
    """Teach the model when *not* to leave the logged nominal prefix.

    This is a learned nominal-preservation objective, not a hard rule.  If the
    nominal prefix is teacher-deployable and no alternative has a material
    teacher advantage, its learned score should dominate.  If an alternative has
    a material deployability/gap advantage, it should outrank nominal.
    """
    if pred_r_dep.numel() <= 1:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    u = torch.nan_to_num(utility.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    trd = teacher_r_dep.float().reshape(-1)
    tro = teacher_r_orc.float().reshape(-1)
    sh = scene_hash.reshape(-1)
    ti = time_index.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    n = min(pr.numel(), pg.numel(), u.numel(), trd.numel(), tro.numel(), sh.numel(), ti.numel(), isn.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, u, trd, tro, sh, ti, isn = pr[:n], pg[:n], u[:n], trd[:n], tro[:n], sh[:n], ti[:n], isn[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(u) & torch.isfinite(trd) & torch.isfinite(tro)
    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_score = trd - float(teacher_gap_weight) * teacher_gap + float(utility_weight) * u
    pred_score = pr - float(pred_gap_weight) * pg + float(utility_weight) * u
    losses: list[torch.Tensor] = []
    keys = torch.stack([sh, ti], dim=1)
    unique = torch.unique(keys[finite], dim=0)
    for key in unique:
        mask = finite & (sh == key[0]) & (ti == key[1])
        if int(mask.sum().item()) < 2:
            continue
        idx = torch.where(mask)[0]
        nom_local = idx[isn[idx]]
        if nom_local.numel() == 0:
            continue
        nom = nom_local[0]
        nom_good = (trd[nom] >= float(nominal_deployable_gamma)) and (teacher_gap[nom] <= float(nominal_gap_max))
        if not bool(nom_good):
            continue
        others = idx[idx != nom]
        if others.numel() == 0:
            continue
        adv = teacher_score[others] - teacher_score[nom]
        should_switch = (adv >= float(teacher_gain_margin)) & (trd[others] >= float(nominal_deployable_gamma))
        should_stay = ~should_switch
        if bool(should_stay.any()):
            diff_stay = pred_score[nom] - pred_score[others[should_stay]]
            losses.append(F.relu(float(margin) - diff_stay).mean())
        if bool(should_switch.any()):
            diff_switch = pred_score[others[should_switch]] - pred_score[nom]
            losses.append(F.relu(float(margin) - diff_switch).mean())
    if not losses:
        return pred_r_dep.sum() * 0.0
    return torch.stack(losses).mean()




def observation_consistent_recovery_advantage_loss(
    pred_r_dep: torch.Tensor,
    pred_gap: torch.Tensor,
    pred_q: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    positive_gain: float = 0.03,
    negative_gain: float = 0.03,
    advantage_margin: float = 0.10,
    regression_weight: float = 1.0,
    ranking_weight: float = 1.0,
    component_inversion_weight: float = 0.5,
    false_positive_weight: float = 0.75,
    nominal_failure_pcd_max: float = 0.20,
    target_min_pred_pcd: float = 0.50,
    nominal_max_pred_pcd: float = 0.48,
    near_weight: float = 1.5,
    contact_weight: float = 1.0,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
) -> torch.Tensor:
    """Observation-consistent counterfactual recovery-advantage calibration.

    Absolute PCD regression is not enough when the learned heads make a *paired
    inversion*: nominal is predicted highly deployable although its teacher PCD
    is near zero, while a shared recovery macro is predicted worse despite being
    the teacher-best action.  This loss operates on complete scene-time candidate
    groups and directly calibrates the counterfactual advantage

        Delta_PCD = PCD(best shared recovery) - PCD(nominal).

    Positive rescue groups receive advantage regression, pairwise ranking, a
    target floor, and component-wise anti-inversion supervision.  Negative rescue
    groups receive an anti-false-positive ranking term.  Bucket-specific weights
    make near-contact calibration explicit instead of inheriting a contact-only
    brake-tail objective.
    """
    if pred_r_dep.numel() <= 1 or not macro_ids or not bucket_ids:
        return pred_r_dep.sum() * 0.0
    pr = torch.nan_to_num(pred_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pg = torch.clamp(torch.nan_to_num(pred_gap.float().reshape(-1), nan=0.0, posinf=20.0, neginf=0.0), min=0.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    pred_drs = _differentiable_shared_success(
        pred_q, root_probs, root_valid, option_valid,
        gamma=success_gamma, temperature=success_temperature,
    ).reshape(-1)
    with torch.no_grad():
        teacher_drs = _differentiable_shared_success(
            teacher_q, root_probs, root_valid, option_valid,
            gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5),
        ).reshape(-1)
    sh, ti = scene_hash.reshape(-1), time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    n = min(pr.numel(), pg.numel(), trd.numel(), tro.numel(), pred_drs.numel(), teacher_drs.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel())
    if n <= 1:
        return pred_r_dep.sum() * 0.0
    pr, pg, trd, tro = pr[:n], pg[:n], trd[:n], tro[:n]
    pred_drs, teacher_drs = pred_drs[:n], teacher_drs[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    finite = torch.isfinite(pr) & torch.isfinite(pg) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(pred_drs) & torch.isfinite(teacher_drs)
    bucket_mask = torch.zeros_like(finite)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == b
    macro_mask = torch.zeros_like(finite)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == m
    finite &= bucket_mask
    if not bool(finite.any()):
        return pred_r_dep.sum() * 0.0

    teacher_gap = torch.clamp(tro - trd, min=0.0)
    teacher_pcd = _torch_pcd_score(teacher_drs, trd, teacher_gap).detach()
    pred_pcd = _torch_pcd_score(pred_drs, pr, pg)
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    keys = torch.stack([sh, ti], dim=1)
    for key in torch.unique(keys[finite], dim=0):
        group = finite & (sh == key[0]) & (ti == key[1])
        idx = torch.where(group)[0]
        if idx.numel() < 2:
            continue
        noms = idx[isn[idx]]
        recs = idx[macro_mask[idx]]
        if noms.numel() == 0 or recs.numel() == 0:
            continue
        nom = noms[0]
        target = recs[torch.argmax(teacher_pcd[recs])]
        teacher_adv = teacher_pcd[target] - teacher_pcd[nom]
        pred_adv = pred_pcd[target] - pred_pcd[nom]
        bw = float(near_weight if int(bid[nom].item()) == 1 else contact_weight)
        group_terms: list[torch.Tensor] = []
        if float(teacher_adv.item()) >= float(positive_gain):
            group_terms.append(float(regression_weight) * F.smooth_l1_loss(pred_adv, teacher_adv))
            group_terms.append(float(ranking_weight) * F.relu(float(advantage_margin) - pred_adv))
            group_terms.append(float(ranking_weight) * 0.5 * F.relu(float(target_min_pred_pcd) - pred_pcd[target]))
            if float(teacher_pcd[nom].item()) <= float(nominal_failure_pcd_max):
                group_terms.append(float(ranking_weight) * 0.5 * F.relu(pred_pcd[nom] - float(nominal_max_pred_pcd)))
            # Anti-inversion on the three deployability components.  The teacher
            # deltas, rather than fixed semantic assumptions, define the sign.
            component = F.smooth_l1_loss(pred_drs[target] - pred_drs[nom], teacher_drs[target] - teacher_drs[nom])
            component = component + F.smooth_l1_loss(pr[target] - pr[nom], trd[target] - trd[nom])
            component = component + 0.5 * F.smooth_l1_loss(pg[nom] - pg[target], teacher_gap[nom] - teacher_gap[target])
            group_terms.append(float(component_inversion_weight) * component)
        elif float(teacher_adv.item()) <= -float(negative_gain):
            group_terms.append(float(false_positive_weight) * F.relu(float(advantage_margin) + pred_adv))
        else:
            # Ambiguous ties still receive gentle advantage regression; this
            # stabilizes the decision boundary without forcing intervention.
            group_terms.append(0.25 * float(regression_weight) * F.smooth_l1_loss(pred_adv, teacher_adv))
        losses.append(torch.stack(group_terms).mean())
        weights.append(bw)
    if not losses:
        return pred_r_dep.sum() * 0.0
    w = torch.as_tensor(weights, dtype=losses[0].dtype, device=losses[0].device)
    stacked = torch.stack(losses)
    return (stacked * w).sum() / w.sum().clamp_min(1.0e-6)



def direct_uncertainty_recovery_value_loss(
    pred_logit: torch.Tensor,
    pred_logvar: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    teacher_r_orc: torch.Tensor,
    teacher_q: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
    option_valid: torch.Tensor,
    scene_hash: torch.Tensor,
    time_index: torch.Tensor,
    macro_type_id: torch.Tensor,
    is_nominal: torch.Tensor,
    bucket_id: torch.Tensor,
    *,
    macro_ids: tuple[int, ...] = (2, 3, 5, 7),
    bucket_ids: tuple[int, ...] = (1, 2),
    temperature: float = 0.12,
    positive_gain: float = 0.03,
    negative_gain: float = 0.02,
    rank_margin: float = 0.04,
    point_weight: float = 0.15,
    listwise_weight: float = 0.35,
    advantage_weight: float = 1.0,
    centered_weight: float = 1.0,
    positive_group_weight: float = 4.0,
    negative_group_weight: float = 1.0,
    ambiguous_group_weight: float = 0.25,
    near_weight: float = 1.5,
    contact_weight: float = 1.0,
    min_group_range: float = 0.01,
    false_positive_weight: float = 1.0,
    variance_floor: float = 0.0025,
    output_mode: str = "probability",
    pairwise_weight: float = 0.0,
    top_rank_weight: float = 0.0,
    success_gamma: float = 0.0,
    success_temperature: float = 0.25,
    pred_opportunity_logit: torch.Tensor | None = None,
    opportunity_weight: float = 0.0,
    opportunity_pos_weight: float = 6.0,
    pred_harm_logit: torch.Tensor | None = None,
    pred_component_harm_logits: torch.Tensor | None = None,
    pred_admission_logit: torch.Tensor | None = None,
    harm_weight: float = 0.0,
    harm_pos_weight: float = 4.0,
    setwise_admission_weight: float = 0.0,
    opportunity_admission_weight: float = 0.35,
    harm_admission_weight: float = 0.75,
    selective_risk_weight: float = 0.0,
    selective_harm_budget: float = 0.05,
    selective_coverage_weight: float = 0.0,
    selective_coverage_target: float = 0.65,
    policy_distill_weight: float = 0.0,
    policy_teacher_temperature: float = 0.08,
    policy_regret_weight: float = 0.0,
    policy_regret_margin: float = 0.0,
    opportunity_soft_label_temperature: float = 0.0,
    harm_soft_label_temperature: float = 0.0,
    policy_decouple_admission: bool = True,
    policy_admission_distill_weight: float = 0.0,
    group_dro_weight: float = 0.0,
    group_dro_temperature: float = 0.35,
    group_dro_severity_thresholds: tuple[float, ...] = (0.25, 0.55),
    teacher_m_star: torch.Tensor | None = None,
    teacher_hard_violation: torch.Tensor | None = None,
    teacher_harm_proxy: torch.Tensor | None = None,
    exact_teacher_pcd: bool = False,
    option_execution_semantics: str = "global",
    pred_rank_logit: torch.Tensor | None = None,
    preference_weight: float = 0.0,
    preference_temperature: float = 0.08,
    preference_min_gap: float = 0.01,
    preference_margin: float = 0.02,
    preference_confidence_scale: float = 0.05,
    preference_regret_weight: float = 0.0,
    preference_listwise_weight: float = 0.0,
    preference_gap_weight: float = 0.0,
    preference_set_weight: float = 0.0,
    preference_set_margin: float = 0.02,
    preference_tie_epsilon_near: float = 0.025,
    preference_tie_epsilon_contact: float = 0.010,
    preference_all_group_set_weight: float = 0.0,
    preference_set_replace_singlewinner: bool = False,
    preference_nominal_margin: float = 0.02,
    preference_harm_margin: float = 0.03,
    preference_set_mass_loss: bool = False,
    preference_noop_nominal_only: bool = False,
    preference_deadzone_margin: float = 0.008,
    preference_conditional_set_weight: float = 0.0,
    preference_conditional_noop_weight: float = 0.35,
    preference_conditional_regret_weight: float = 0.5,
    preference_conditional_pairwise_weight: float = 0.0,
    preference_conditional_pairwise_min_gap: float = 0.01,
    preference_conditional_pairwise_margin: float = 0.02,
    preference_proposal_topk_weight: float = 0.0,
    preference_proposal_topk: int = 3,
    preference_proposal_margin: float = 0.02,
    delta_nll_weight: float = 0.0,
    delta_sign_weight: float = 0.0,
    delta_sign_temperature: float = 0.04,
    certificate_policy_top1_weight: float = 0.0,
    certificate_policy_top1_sign_weight: float = 0.0,
    certificate_policy_top1_temperature: float = 0.04,
    ordinal_evidence_policy_top1_weight: float = 0.0,
    ordinal_evidence_all_candidate_weight: float = 0.0,
    ordinal_evidence_focal_gamma: float = 1.5,
    ordinal_evidence_ordered_nll_top1_weight: float = 0.0,
    ordinal_evidence_ordered_nll_all_weight: float = 0.0,
    ordinal_evidence_harm_class_weight: float = 2.0,
    ordinal_evidence_dead_class_weight: float = 0.5,
    ordinal_evidence_benefit_class_weight: float = 1.25,
    ordinal_evidence_hard_harm_weight: float = 0.0,
    ordinal_evidence_hard_benefit_weight: float = 0.0,
    ordinal_evidence_hard_example_gamma: float = 2.0,
    ordinal_evidence_class_balanced_weight: float = 0.0,
    ordinal_evidence_batch_balanced: bool = False,
    ordinal_evidence_independent_tails: bool = False,
    ordinal_evidence_factorized_harm: bool = False,
    ordinal_evidence_factorized_harm_temperature: float = 0.05,
    ordinal_evidence_factorized_harm_drs_tolerance: float = 0.05,
    ordinal_evidence_factorized_harm_dep_tolerance: float = 0.05,
    ordinal_evidence_factorized_harm_gap_tolerance: float = 0.05,
    ordinal_evidence_factorized_harm_hard_tolerance: float = 0.05,
    ordinal_evidence_factorized_harm_proxy_tolerance: float = 0.05,
    ordinal_evidence_dep_boundary_aligned: bool = False,
    ordinal_evidence_gap_ordinal_only: bool = False,
    ordinal_evidence_component_tail_weight: float = 0.0,
    ordinal_evidence_component_margin_regression_weight: float = 0.0,
    ordinal_evidence_component_margin_target_mode: str = "raw",
    ordinal_evidence_component_margin_target_scale: float = 0.10,
    ordinal_evidence_component_margin_canonical_scales: str | tuple[float, ...] = "",
    ordinal_evidence_component_margin_regression_reliability: str | tuple[float, ...] = "",
    ordinal_evidence_component_underestimation_weight: float = 0.0,
    ordinal_evidence_safe_positive_component_overestimation_weight: float = 0.0,
    ordinal_evidence_benefit_margin_regression_weight: float = 0.0,
    ordinal_evidence_benefit_margin_temperature: float = 0.025,
    ordinal_evidence_joint_reserve_regression_weight: float = 0.0,
    ordinal_evidence_joint_reserve_boundary_weight: float = 0.0,
    ordinal_evidence_joint_reserve_boundary_width: float = 0.05,
    ordinal_evidence_component_reliability: str | tuple[float, ...] = "",
    ordinal_evidence_global_balance: bool = False,
    ordinal_evidence_safe_set_temperature: float = 0.05,
    ordinal_evidence_safe_benefit_target: bool = False,
    ordinal_evidence_group_opportunity_weight: float = 0.0,
    ordinal_evidence_admission_weight: float = 0.0,
    ordinal_evidence_admission_pos_weight: float = 4.0,
    ordinal_evidence_admission_harm_negative_weight: float = 2.0,
    ordinal_evidence_balanced_replaces_erm: bool = False,
    ordinal_evidence_benefit_margin_weight: float = 0.0,
    ordinal_evidence_harm_margin_weight: float = 0.0,
    ordinal_evidence_target_probability: float = 0.60,
    evidence_calibrator_residual: torch.Tensor | None = None,
    evidence_calibrator_anchor_weight: float = 0.0,
    ordinal_evidence_proposal_topk_weight: float = 0.0,
    ordinal_evidence_proposal_topk: int = 3,
    ordinal_evidence_proposal_rank_decay: float = 0.75,
    ordinal_evidence_intragroup_benefit_weight: float = 0.0,
    ordinal_evidence_intragroup_harm_weight: float = 0.0,
    ordinal_evidence_benefit_listwise_weight: float = 0.0,
    ordinal_evidence_benefit_listwise_temperature: float = 0.08,
    ordinal_evidence_safe_utility_regression_weight: float = 0.0,
    ordinal_evidence_safe_utility_listwise_weight: float = 0.0,
    ordinal_evidence_safe_utility_temperature: float = 0.10,
    ordinal_evidence_eligible_policy_weight: float = 0.0,
    ordinal_evidence_eligible_policy_temperature: float = 0.10,
    ordinal_evidence_eligibility_logit_temperature: float = 0.25,
    ordinal_evidence_eligible_opportunity_threshold: float = 0.65,
    ordinal_evidence_eligible_harm_threshold: float = 0.30,
    ordinal_evidence_eligibility_boundary_weight: float = 0.0,
    ordinal_evidence_eligibility_boundary_margin: float = 0.20,
    ordinal_evidence_frontier_pairwise_weight: float = 0.0,
    ordinal_evidence_frontier_pairwise_margin: float = 0.25,
    ordinal_evidence_safe_hard_negative_weight: float = 0.0,
    ordinal_evidence_safe_hard_negative_margin: float = 0.05,
    ordinal_evidence_safe_hard_negative_teacher_scale: float = 0.0,
    ordinal_evidence_categorical_group_policy: bool = False,
    ordinal_evidence_intragroup_margin: float = 0.25,
    ordinal_evidence_pairwise_benefit_weight: float = 0.0,
    ordinal_evidence_pairwise_harm_weight: float = 0.0,
    ordinal_evidence_pairwise_margin: float = 0.25,
    strict_shape_contract: bool = False,
    pred_delta_mean: torch.Tensor | None = None,
    pred_delta_logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Learn a tri-state, nominal-relative, policy-level recovery certificate.

    v40 used an absolute heteroscedastic PCD target.  Positive recovery groups
    are rare, so that objective is dominated by easy negative/tied groups and a
    frozen scene CLS can converge to an almost constant prediction.  v41 keeps a
    small absolute anchor but makes the primary target the scene-time centred
    advantage ``PCD(candidate)-PCD(nominal)``.  Positive and negative groups are
    balanced explicitly, and listwise supervision is skipped for true ties.

    The variance output remains available for diagnostics/backward compatibility;
    selection uses a validation-calibrated additive residual bound instead of
    trusting the model's self-reported standard deviation.
    """
    # Keep an autograd connection to every optional direct branch.  Staged
    # training intentionally freezes most heads and some perfectly valid group
    # batches contain no supervision for the currently trainable head (for
    # example, a preference-only batch with no positive recovery opportunity).
    # In that case the mathematical loss is exactly zero, but returning a zero
    # made only from a frozen value tensor produces ``requires_grad=False`` and
    # makes ``backward()`` fail.  Adding this zero-valued anchor preserves the
    # objective and optimizer behaviour while turning such batches into proper
    # zero-gradient no-ops.
    grad_anchor = pred_logit.float().sum() * 0.0
    for optional_output in (
        pred_logvar,
        pred_opportunity_logit,
        pred_harm_logit,
        pred_component_harm_logits,
        pred_admission_logit,
        pred_rank_logit,
        pred_delta_mean,
        pred_delta_logvar,
        evidence_calibrator_residual,
    ):
        if optional_output is not None:
            grad_anchor = grad_anchor + optional_output.float().sum() * 0.0

    if pred_logit.numel() <= 1:
        return grad_anchor
    raw_score = pred_logit.float().reshape(-1)
    mode = str(output_mode or "probability").strip().lower()
    if mode not in {"probability", "score"}:
        raise ValueError(f"Unsupported direct value output mode: {output_mode!r}")
    # v42 OCSAVA predicts an unbounded preference score. Only the optional
    # absolute anchor is mapped through sigmoid; pairwise/listwise objectives
    # operate on the raw score so candidate advantages are not compressed.
    score = raw_score if mode == "score" else torch.sigmoid(raw_score)
    rank_score = score if pred_rank_logit is None else pred_rank_logit.float().reshape(-1)
    direct_delta = None if pred_delta_mean is None else pred_delta_mean.float().reshape(-1)
    direct_delta_logvar = None if pred_delta_logvar is None else pred_delta_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    component_harm_logits = None
    if pred_component_harm_logits is not None:
        component_harm_logits = pred_component_harm_logits.float()
        if component_harm_logits.ndim != 2 or component_harm_logits.shape[-1] < 3:
            raise ValueError(
                "pred_component_harm_logits must have shape [N, >=3]; "
                "v48.27 uses five DRS/deployability/gap/hard/harm-proxy components"
            )
    point_mean = torch.sigmoid(raw_score)
    logvar = pred_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    trd = torch.nan_to_num(teacher_r_dep.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(teacher_r_orc.float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    teacher_hard = (
        torch.zeros_like(trd)
        if teacher_hard_violation is None
        else torch.nan_to_num(teacher_hard_violation.float().reshape(-1), nan=0.0, posinf=10.0, neginf=0.0)
    )
    teacher_harm = (
        torch.zeros_like(trd)
        if teacher_harm_proxy is None
        else torch.nan_to_num(teacher_harm_proxy.float().reshape(-1), nan=0.0, posinf=10.0, neginf=0.0)
    )
    with torch.no_grad():
        if bool(exact_teacher_pcd):
            if teacher_m_star is None:
                raise ValueError("exact_teacher_pcd=true requires teacher_m_star")
            teacher_drs = _exact_teacher_recovery_success(
                teacher_q, teacher_m_star, root_probs, root_valid, option_valid,
                gamma=success_gamma, semantics=option_execution_semantics,
            ).reshape(-1)
        else:
            teacher_drs = _recovery_success_proxy(
                teacher_q, root_probs, root_valid, option_valid,
                gamma=success_gamma, temperature=max(0.08, success_temperature * 0.5),
                semantics=option_execution_semantics,
            ).reshape(-1)
    sh, ti = scene_hash.reshape(-1), time_index.reshape(-1)
    mac = macro_type_id.reshape(-1)
    isn = is_nominal.float().reshape(-1) > 0.5
    bid = bucket_id.reshape(-1)
    sizes = [score.numel(), rank_score.numel(), point_mean.numel(), logvar.numel(), trd.numel(), tro.numel(), teacher_drs.numel(), teacher_hard.numel(), teacher_harm.numel(), sh.numel(), ti.numel(), mac.numel(), isn.numel(), bid.numel()]
    if direct_delta is not None:
        sizes.append(direct_delta.numel())
    if direct_delta_logvar is not None:
        sizes.append(direct_delta_logvar.numel())
    if component_harm_logits is not None:
        sizes.append(component_harm_logits.shape[0])
    if pred_admission_logit is not None:
        sizes.append(pred_admission_logit.numel())
    if bool(strict_shape_contract) and len(set(int(x) for x in sizes)) != 1:
        raise ValueError(
            "direct recovery loss shape contract violated: "
            + ", ".join(str(int(x)) for x in sizes)
        )
    n = min(sizes)
    score, rank_score, point_mean, logvar, trd, tro, teacher_drs = score[:n], rank_score[:n], point_mean[:n], logvar[:n], trd[:n], tro[:n], teacher_drs[:n]
    teacher_hard, teacher_harm = teacher_hard[:n], teacher_harm[:n]
    sh, ti, mac, isn, bid = sh[:n], ti[:n], mac[:n], isn[:n], bid[:n]
    if direct_delta is not None:
        direct_delta = direct_delta[:n]
    if direct_delta_logvar is not None:
        direct_delta_logvar = direct_delta_logvar[:n]
    if component_harm_logits is not None:
        component_harm_logits = component_harm_logits[:n]
    bucket_mask = torch.zeros((n,), dtype=torch.bool, device=score.device)
    for b in tuple(int(x) for x in bucket_ids):
        bucket_mask |= bid == b
    finite = bucket_mask & torch.isfinite(score) & torch.isfinite(rank_score) & torch.isfinite(point_mean) & torch.isfinite(logvar) & torch.isfinite(trd) & torch.isfinite(tro) & torch.isfinite(teacher_drs) & torch.isfinite(teacher_hard) & torch.isfinite(teacher_harm)
    if not bool(finite.any()):
        return grad_anchor
    teacher_gap_vector = torch.clamp(tro - trd, min=0.0)
    target = _torch_pcd_score(teacher_drs, trd, teacher_gap_vector).detach().clamp(0.0, 1.0)
    variance = torch.exp(logvar).clamp_min(float(variance_floor))
    point = 0.5 * ((point_mean - target) ** 2 / variance + torch.log(variance))
    point_loss = point[finite].mean()

    macro_mask = torch.zeros((n,), dtype=torch.bool, device=score.device)
    for m in tuple(int(x) for x in macro_ids):
        macro_mask |= mac == m
    # v44: include regime in the group key; Near and Contact may share the
    # same WOMD scene-time but have different pressure-future teacher targets.
    keys = torch.stack([bid, sh, ti], dim=1)
    tau = max(float(temperature), 1.0e-3)
    opportunity_logits = None
    if pred_opportunity_logit is not None:
        opportunity_logits = pred_opportunity_logit.float().reshape(-1)[:n]
    harm_logits = None
    if pred_harm_logit is not None:
        harm_logits = pred_harm_logit.float().reshape(-1)[:n]
    admission_logits = None
    if pred_admission_logit is not None:
        admission_logits = pred_admission_logit.float().reshape(-1)[:n]
    group_losses: list[torch.Tensor] = []
    group_weights: list[float] = []
    group_domains: list[tuple[int, int, int, int]] = []
    # v48.12 TRIDENT: collect frozen-policy top-1 evidence across groups so
    # benefit and harm ranking are optimized directly, not only through local
    # class likelihoods.  Regime-specific pairwise losses target the AUCs used
    # by the Natural gate and are especially important for the harmful tail.
    evidence_policy_records: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
    # v48.17 BRIDGE: class balance must be enforced across proposal candidates
    # in the whole minibatch (and separately by regime), not independently in
    # each scene-time group.  Most groups contain a single evidence class, so
    # v48.16's within-group averaging reduced to another dead-zone-dominated NLL.
    evidence_proposal_records: list[tuple] = []
    for key in torch.unique(keys[finite], dim=0):
        idx = torch.where(finite & (bid == key[0]) & (sh == key[1]) & (ti == key[2]))[0]
        noms = idx[isn[idx]]
        recs = idx[macro_mask[idx] & (~isn[idx])]
        if bool(strict_shape_contract) and noms.numel() != 1:
            raise ValueError(
                f"direct recovery group contract requires exactly one nominal; "
                f"key={tuple(int(x) for x in key.tolist())} nominal_count={int(noms.numel())}"
            )
        if noms.numel() == 0 or recs.numel() == 0:
            continue
        nom = noms[0]
        t_delta = target[recs] - target[nom]
        factorized_harm_margin = None
        factorized_harm_target = None
        factorized_harm_binary = None
        factorized_component_margins = None
        factorized_component_targets = None
        component_harm_delta_logits = None
        if bool(ordinal_evidence_factorized_harm):
            tolerances = ComponentVetoTolerances(
                drs=float(ordinal_evidence_factorized_harm_drs_tolerance),
                deployability_gate=float(ordinal_evidence_factorized_harm_dep_tolerance),
                gap_discount=float(ordinal_evidence_factorized_harm_gap_tolerance),
                hard_violation=float(ordinal_evidence_factorized_harm_hard_tolerance),
                harm_proxy=float(ordinal_evidence_factorized_harm_proxy_tolerance),
                deployability_boundary_aligned=bool(ordinal_evidence_dep_boundary_aligned),
                gap_ordinal_only=bool(ordinal_evidence_gap_ordinal_only),
            )
            factorized_component_margins = component_veto_terms_torch(
                candidate_drs=teacher_drs[recs],
                nominal_drs=teacher_drs[nom].expand_as(teacher_drs[recs]),
                candidate_r_dep=trd[recs],
                nominal_r_dep=trd[nom].expand_as(trd[recs]),
                candidate_gap=teacher_gap_vector[recs],
                nominal_gap=teacher_gap_vector[nom].expand_as(teacher_gap_vector[recs]),
                candidate_hard=teacher_hard[recs],
                nominal_hard=teacher_hard[nom].expand_as(teacher_hard[recs]),
                candidate_harm_proxy=teacher_harm[recs],
                nominal_harm_proxy=teacher_harm[nom].expand_as(teacher_harm[recs]),
                tolerances=tolerances,
            ).detach()
            factorized_harm_margin = factorized_component_margins.max(dim=-1).values
            component_count = (
                min(int(component_harm_logits.shape[-1]), int(factorized_component_margins.shape[-1]))
                if component_harm_logits is not None else 0
            )
            factorized_component_targets = (
                component_veto_soft_target(
                    factorized_component_margins[:, :component_count],
                    temperature=float(ordinal_evidence_factorized_harm_temperature),
                ).detach()
                if component_count > 0 else None
            )
            if component_harm_logits is not None and component_count > 0:
                component_harm_delta_logits = (
                    component_harm_logits[recs, :component_count]
                    - component_harm_logits[nom, :component_count].unsqueeze(0)
                )
            factorized_harm_margin_check = component_veto_margin_torch(
                candidate_drs=teacher_drs[recs],
                nominal_drs=teacher_drs[nom].expand_as(teacher_drs[recs]),
                candidate_r_dep=trd[recs],
                nominal_r_dep=trd[nom].expand_as(trd[recs]),
                candidate_gap=teacher_gap_vector[recs],
                nominal_gap=teacher_gap_vector[nom].expand_as(teacher_gap_vector[recs]),
                candidate_hard=teacher_hard[recs],
                nominal_hard=teacher_hard[nom].expand_as(teacher_hard[recs]),
                candidate_harm_proxy=teacher_harm[recs],
                nominal_harm_proxy=teacher_harm[nom].expand_as(teacher_harm[recs]),
                tolerances=tolerances,
            ).detach()
            if not torch.allclose(factorized_harm_margin, factorized_harm_margin_check):
                raise RuntimeError("component-veto scalar/vector target drift")
            factorized_harm_target = component_veto_soft_target(
                factorized_harm_margin,
                temperature=float(ordinal_evidence_factorized_harm_temperature),
            ).detach()
            factorized_harm_binary = factorized_harm_margin > 0.0
        p_delta = score[recs] - score[nom]
        if direct_delta is not None:
            p_delta = direct_delta[recs]
        r_delta = rank_score[recs] - rank_score[nom]
        # Emphasise decision-relevant deltas while retaining all candidates.
        mag_w = 1.0 + torch.clamp(t_delta.abs() / max(float(positive_gain), 1.0e-3), max=4.0)
        centered = (mag_w * F.smooth_l1_loss(p_delta, t_delta, reduction='none')).sum() / mag_w.sum().clamp_min(1.0e-6)
        terms = [float(centered_weight) * centered]
        if float(delta_nll_weight) > 0.0:
            if direct_delta_logvar is not None:
                delta_var = torch.exp(direct_delta_logvar[recs]).clamp_min(float(variance_floor))
            else:
                delta_var = (variance[recs] + variance[nom]).clamp_min(float(variance_floor))
            delta_nll = 0.5 * (((p_delta - t_delta).square() / delta_var) + torch.log(delta_var))
            terms.append(float(delta_nll_weight) * (mag_w * delta_nll).sum() / mag_w.sum().clamp_min(1.0e-6))
        if float(delta_sign_weight) > 0.0:
            sign_tau = max(float(delta_sign_temperature), 1.0e-4)
            positive_soft = torch.sigmoid((t_delta - float(positive_gain)) / sign_tau).detach()
            harmful_soft = torch.sigmoid((-float(negative_gain) - t_delta) / sign_tau).detach()
            positive_bce = F.binary_cross_entropy_with_logits(p_delta / sign_tau, positive_soft)
            harmful_bce = F.binary_cross_entropy_with_logits(-p_delta / sign_tau, harmful_soft)
            terms.append(float(delta_sign_weight) * 0.5 * (positive_bce + harmful_bce))
        # v48.9 PACER: Stage C is evaluated only on the candidate selected
        # by the frozen Stage-P preference policy.  Train the certificate on
        # that induced distribution instead of letting thousands of unused
        # candidates dominate the relative-gain regression.
        if (
            direct_delta is not None
            and (float(certificate_policy_top1_weight) > 0.0
                 or float(certificate_policy_top1_sign_weight) > 0.0)
        ):
            policy_j = int(torch.argmax(r_delta.detach()).item())
            policy_pred = p_delta[policy_j]
            policy_teacher = t_delta[policy_j].detach()
            if float(certificate_policy_top1_weight) > 0.0:
                policy_reg = F.smooth_l1_loss(policy_pred, policy_teacher, reduction="mean")
                terms.append(float(certificate_policy_top1_weight) * policy_reg)
            if float(certificate_policy_top1_sign_weight) > 0.0:
                policy_tau = max(float(certificate_policy_top1_temperature), 1.0e-4)
                policy_positive = torch.sigmoid(
                    (policy_teacher - float(positive_gain)) / policy_tau
                ).detach()
                policy_harmful = torch.sigmoid(
                    (-float(negative_gain) - policy_teacher) / policy_tau
                ).detach()
                policy_pos_bce = F.binary_cross_entropy_with_logits(
                    policy_pred / policy_tau, policy_positive
                )
                policy_harm_bce = F.binary_cross_entropy_with_logits(
                    -policy_pred / policy_tau, policy_harmful
                )
                terms.append(
                    0.5 * float(certificate_policy_top1_sign_weight)
                    * (policy_pos_bce + policy_harm_bce)
                )
        group_range = float((target[idx].max() - target[idx].min()).item())
        if group_range >= float(min_group_range) and float(listwise_weight) > 0.0:
            teacher_prob = torch.softmax((target[idx] - target[idx].mean()) / tau, dim=0)
            pred_log_prob = torch.log_softmax((score[idx] - score[idx].mean()) / tau, dim=0)
            terms.append(float(listwise_weight) * F.kl_div(pred_log_prob, teacher_prob, reduction='sum'))
        best_j = int(torch.argmax(t_delta).item())
        t_adv = t_delta[best_j]
        p_adv = p_delta[best_j]
        pos_mask = t_delta >= float(positive_gain)
        harmful_mask = t_delta <= -float(negative_gain)
        # Only deltas beyond the negative margin are harmful negatives. Exact
        # and near ties form the tri-state dead zone and should be regressed
        # gently toward their teacher delta rather than forced below a margin.
        neg_mask = harmful_mask
        tie_mask = (~pos_mask) & (~harmful_mask)

        # v48.10 COPE: conditional option preference.  Preference answers
        # "which recovery option is best, conditional on intervening" and is
        # therefore trained only over recovery candidates.  Nominal admission is
        # handled by the separate ordinal evidence certificate.  This prevents
        # abundant no-op groups from turning Stage P into another gate classifier.
        if float(preference_conditional_set_weight) > 0.0:
            pref_tau = max(float(preference_temperature), 1.0e-3)
            tie_eps = (
                float(preference_tie_epsilon_near)
                if int(bid[nom].item()) == 1
                else float(preference_tie_epsilon_contact)
            )
            best_recovery_teacher = t_delta.max().detach()
            acceptable_recovery = t_delta >= (best_recovery_teacher - tie_eps)
            conditional_logits = r_delta / pref_tau
            set_mass = torch.logsumexp(conditional_logits, dim=0) - torch.logsumexp(
                conditional_logits[acceptable_recovery], dim=0
            )
            rejected_recovery = ~acceptable_recovery
            if bool(rejected_recovery.any()):
                accept_score = torch.logsumexp(
                    conditional_logits[acceptable_recovery], dim=0
                ) * pref_tau
                reject_score = torch.logsumexp(
                    conditional_logits[rejected_recovery], dim=0
                ) * pref_tau
                set_mass = set_mass + F.softplus(
                    (float(preference_set_margin) - (accept_score - reject_score)) / pref_tau
                ) * pref_tau
            teacher_prob_cond = torch.softmax(t_delta.detach() / pref_tau, dim=0)
            pred_prob_cond = torch.softmax(conditional_logits, dim=0)
            expected_teacher_cond = (pred_prob_cond * t_delta.detach()).sum()
            conditional_regret = torch.clamp(
                best_recovery_teacher - expected_teacher_cond, min=0.0
            )
            material_weight = (
                1.0
                if float(best_recovery_teacher.item()) >= float(positive_gain)
                else max(0.0, float(preference_conditional_noop_weight))
            )
            conditional_term = set_mass
            if float(preference_conditional_regret_weight) > 0.0:
                conditional_term = conditional_term + float(preference_conditional_regret_weight) * conditional_regret.square()
            # A low-weight teacher distribution term stabilizes close recovery
            # options without imposing a uniform ordering inside the acceptable set.
            conditional_term = conditional_term + 0.15 * F.kl_div(
                torch.log_softmax(conditional_logits, dim=0),
                teacher_prob_cond, reduction="sum"
            )
            # v48.12 TRIDENT recovery-pair tournament.  Set likelihood is
            # intentionally indifferent inside the teacher-equivalent set, but
            # it provides weak gradients for materially ordered recovery pairs.
            # Direct gap-weighted comparisons improve the actual group top-1
            # objective without inventing an ordering for near ties.
            if float(preference_conditional_pairwise_weight) > 0.0 and r_delta.numel() > 1:
                teacher_pair_gap = t_delta.detach().unsqueeze(1) - t_delta.detach().unsqueeze(0)
                ordered_pairs = teacher_pair_gap >= float(preference_conditional_pairwise_min_gap)
                ordered_pairs.fill_diagonal_(False)
                if bool(ordered_pairs.any()):
                    pred_pair_gap = r_delta.unsqueeze(1) - r_delta.unsqueeze(0)
                    pair_conf = torch.clamp(
                        teacher_pair_gap[ordered_pairs]
                        / max(float(preference_confidence_scale), 1.0e-4),
                        0.0, 1.0,
                    )
                    required_pair_gap = (
                        float(preference_conditional_pairwise_margin)
                        + torch.clamp(teacher_pair_gap[ordered_pairs], max=0.20)
                    )
                    pair_loss = F.softplus(
                        (required_pair_gap - pred_pair_gap[ordered_pairs]) / pref_tau
                    ) * pref_tau
                    conditional_term = conditional_term + float(preference_conditional_pairwise_weight) * (
                        (pair_conf * pair_loss).sum() / pair_conf.sum().clamp_min(1.0e-6)
                    )
            # v48.13 TERRA proposal-set supervision. Exact top-1 labels are
            # unstable in Near and weakly identified in Contact. Require at least
            # one teacher-acceptable recovery to enter the model top-k proposal.
            if float(preference_proposal_topk_weight) > 0.0 and r_delta.numel() > 1:
                proposal_k = min(max(1, int(preference_proposal_topk)), int(r_delta.numel()))
                proposal_boundary = torch.topk(r_delta, k=proposal_k).values[-1]
                acceptable_anchor = torch.logsumexp(r_delta[acceptable_recovery] / pref_tau, dim=0) * pref_tau
                proposal_loss = F.softplus(
                    (float(preference_proposal_margin) + proposal_boundary - acceptable_anchor) / pref_tau
                ) * pref_tau
                conditional_term = conditional_term + float(preference_proposal_topk_weight) * proposal_loss
            terms.append(
                float(preference_conditional_set_weight) * material_weight * conditional_term
            )

        # v48.8 SCOPE: conflict-free nominal-inclusive set preference.  v48.7
        # simultaneously treated near-tied candidates as both equivalent (set
        # KL) and as ordered best-vs-rest competitors.  This objective replaces
        # the single-winner family when requested and supervises every group:
        # material-recovery groups prefer a teacher-equivalent recovery set;
        # no-opportunity groups prefer nominal plus only genuinely dead-zone
        # recoveries, while harmful candidates are pushed below nominal.
        if float(preference_all_group_set_weight) > 0.0:
            pref_tau = max(float(preference_temperature), 1.0e-3)
            tie_eps = (
                float(preference_tie_epsilon_near)
                if int(bid[nom].item()) == 1
                else float(preference_tie_epsilon_contact)
            )
            all_teacher = torch.cat([t_delta.new_zeros(1), t_delta], dim=0)
            all_pred = torch.cat([r_delta.new_zeros(1), r_delta], dim=0)
            best_teacher_all = all_teacher.max().detach()
            material_positive = bool(float(t_delta.max().item()) >= float(positive_gain))
            if material_positive:
                acceptable_all = all_teacher >= (best_teacher_all - tie_eps)
                acceptable_all[0] = False
            elif bool(preference_noop_nominal_only):
                # When no material recovery exists, the policy target is the
                # nominal action.  Dead-zone recoveries may be harmless, but
                # executing them is still an unnecessary intervention and was
                # the main source of v48.8 false switches.
                acceptable_all = torch.zeros_like(all_teacher, dtype=torch.bool)
                acceptable_all[0] = True
            else:
                acceptable_all = all_teacher >= -tie_eps
                acceptable_all[0] = True
            if bool(acceptable_all.any()):
                logits = all_pred / pref_tau
                if bool(preference_set_mass_loss):
                    # Partial-label set likelihood: reward probability mass on
                    # the acceptable set without forcing its members to have
                    # identical logits.  The old uniform-target KL imposed an
                    # artificial within-set ordering constraint.
                    set_term = torch.logsumexp(logits, dim=0) - torch.logsumexp(
                        logits[acceptable_all], dim=0
                    )
                else:
                    target_set = acceptable_all.to(dtype=all_pred.dtype)
                    target_set = target_set / target_set.sum().clamp_min(1.0)
                    pred_log = torch.log_softmax(logits, dim=0)
                    set_term = F.kl_div(pred_log, target_set, reduction="sum")
                rejected_all = ~acceptable_all
                if bool(rejected_all.any()):
                    accept_score = torch.logsumexp(logits[acceptable_all], dim=0) * pref_tau
                    reject_score = torch.logsumexp(logits[rejected_all], dim=0) * pref_tau
                    required_margin = (
                        float(preference_set_margin)
                        if material_positive else float(preference_nominal_margin)
                    )
                    set_term = set_term + F.softplus(
                        (required_margin - (accept_score - reject_score)) / pref_tau
                    ) * pref_tau
                if (not material_positive) and bool(preference_noop_nominal_only) and bool(tie_mask.any()):
                    # Dead-zone candidates receive only a weak intervention-cost
                    # margin; they are not mislabeled as harmful.
                    dead_rank = r_delta[tie_mask]
                    set_term = set_term + 0.25 * (
                        F.softplus((dead_rank + float(preference_deadzone_margin)) / pref_tau)
                        * pref_tau
                    ).mean()
                expected_teacher = (torch.softmax(logits, dim=0) * all_teacher).sum()
                set_regret = torch.clamp(best_teacher_all - expected_teacher, min=0.0)
                confidence = (
                    torch.clamp(
                        (best_teacher_all - float(positive_gain))
                        / max(float(preference_confidence_scale), 1.0e-4),
                        0.0, 1.0,
                    )
                    if material_positive else all_pred.new_tensor(1.0)
                )
                terms.append(
                    float(preference_all_group_set_weight)
                    * confidence
                    * (set_term + 0.5 * set_regret.square())
                )
            if bool(harmful_mask.any()) and float(preference_harm_margin) > 0.0:
                harm_rank = r_delta[harmful_mask]
                harm_penalty = F.softplus(
                    (harm_rank + float(preference_harm_margin)) / pref_tau
                ) * pref_tau
                terms.append(
                    0.5 * float(preference_all_group_set_weight) * harm_penalty.mean()
                )

        # v48.5 ECPR: confidence-paced exact best-vs-rest preference.  It is
        # applied only when the exact teacher has a material recovery opportunity,
        # and downweights near-ties whose ordering is not stable across splits.
        if (not bool(preference_set_replace_singlewinner)) and bool(pos_mask.any()) and (float(preference_weight) > 0.0 or float(preference_regret_weight) > 0.0):
            best_rank_j = int(torch.argmax(t_delta).item())
            best_teacher = t_delta[best_rank_j]
            gaps = (best_teacher - t_delta).detach()
            competitor = torch.arange(recs.numel(), device=recs.device) != best_rank_j
            competitor &= gaps >= float(preference_min_gap)
            pref_tau = max(float(preference_temperature), 1.0e-3)
            if bool(competitor.any()) and float(preference_weight) > 0.0:
                confidence = torch.clamp(gaps[competitor] / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                pred_gap = r_delta[best_rank_j] - r_delta[competitor]
                required = float(preference_margin) + torch.clamp(gaps[competitor], max=0.25)
                pref = F.softplus((required - pred_gap) / pref_tau) * pref_tau
                terms.append(float(preference_weight) * (confidence * pref).sum() / confidence.sum().clamp_min(1.0e-6))
            # Exact expected-regret term over nominal plus recovery candidates.
            if float(preference_regret_weight) > 0.0:
                pref_logits = torch.cat([r_delta.new_zeros(1), r_delta], dim=0) / pref_tau
                pref_prob = torch.softmax(pref_logits, dim=0)
                teacher_util_pref = torch.cat([t_delta.new_zeros(1), t_delta], dim=0)
                expected = (pref_prob * teacher_util_pref).sum()
                regret_pref = torch.clamp(best_teacher - expected, min=0.0)
                confidence_group = torch.clamp((best_teacher - float(positive_gain)) / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(float(preference_regret_weight) * confidence_group * regret_pref.square())
            if float(preference_listwise_weight) > 0.0:
                teacher_pref = torch.softmax(t_delta.detach() / pref_tau, dim=0)
                pred_pref_log = torch.log_softmax(r_delta / pref_tau, dim=0)
                confidence_group = torch.clamp((best_teacher - float(positive_gain)) / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(
                    float(preference_listwise_weight)
                    * confidence_group
                    * F.kl_div(pred_pref_log, teacher_pref, reduction="sum")
                )
            if float(preference_gap_weight) > 0.0 and recs.numel() > 1:
                teacher_others = t_delta[torch.arange(recs.numel(), device=recs.device) != best_rank_j]
                teacher_second = teacher_others.max()
                teacher_best_gap = (best_teacher - teacher_second).detach().clamp(min=0.0, max=0.25)
                pred_others = r_delta[torch.arange(recs.numel(), device=recs.device) != best_rank_j]
                pred_alt = torch.maximum(pred_others.max(), r_delta.new_zeros(()))
                pred_best_gap = r_delta[best_rank_j] - pred_alt
                confidence_gap = torch.clamp(teacher_best_gap / max(float(preference_confidence_scale), 1.0e-4), 0.0, 1.0)
                terms.append(
                    float(preference_gap_weight)
                    * confidence_gap
                    * F.smooth_l1_loss(pred_best_gap, teacher_best_gap, reduction="mean")
                )
            # v48.7 SPIRE: Near-contact often contains several teacher-equivalent
            # recovery candidates.  Forcing an arbitrary single winner creates
            # cross-seed label flips.  Learn an acceptable *set* within a
            # regime-specific exact-PCD epsilon, while still separating that set
            # from nominal and materially worse recovery candidates.
            if float(preference_set_weight) > 0.0:
                tie_eps = (
                    float(preference_tie_epsilon_near)
                    if int(bid[nom].item()) == 1
                    else float(preference_tie_epsilon_contact)
                )
                acceptable = t_delta >= (best_teacher - tie_eps)
                unacceptable = ~acceptable
                if bool(acceptable.any()):
                    set_target = acceptable.to(dtype=r_delta.dtype)
                    set_target = set_target / set_target.sum().clamp_min(1.0)
                    pred_set_log = torch.log_softmax(r_delta / pref_tau, dim=0)
                    set_kl = F.kl_div(pred_set_log, set_target, reduction="sum")
                    set_conf = torch.clamp(
                        (best_teacher - float(positive_gain))
                        / max(float(preference_confidence_scale), 1.0e-4),
                        0.0, 1.0,
                    )
                    set_term = set_kl
                    if bool(unacceptable.any()):
                        accept_score = torch.logsumexp(r_delta[acceptable] / pref_tau, dim=0) * pref_tau
                        reject_score = torch.maximum(
                            torch.logsumexp(r_delta[unacceptable] / pref_tau, dim=0) * pref_tau,
                            r_delta.new_zeros(()),
                        )
                        set_term = set_term + F.softplus(
                            (float(preference_set_margin) - (accept_score - reject_score)) / pref_tau
                        ) * pref_tau
                    terms.append(float(preference_set_weight) * set_conf * set_term)
        opp_delta_logits = None
        if opportunity_logits is not None:
            opp_delta_logits = opportunity_logits[recs] - opportunity_logits[nom]
        harm_delta_logits = None
        if harm_logits is not None:
            harm_delta_logits = harm_logits[recs] - harm_logits[nom]
        admission_delta_logits = None
        if admission_logits is not None:
            admission_delta_logits = admission_logits[recs] - admission_logits[nom]
        if opp_delta_logits is not None and float(opportunity_weight) > 0.0:
            opp_temp = float(opportunity_soft_label_temperature)
            labels = (
                torch.sigmoid((t_delta - float(positive_gain)) / max(opp_temp, 1.0e-4))
                if opp_temp > 0.0 else pos_mask.to(dtype=opp_delta_logits.dtype)
            ).to(dtype=opp_delta_logits.dtype)
            # Opportunity is explicitly candidate-vs-nominal, matching the
            # deployed admission decision rather than an absolute candidate tag.
            # Soft labels avoid treating tiny cross-split teacher perturbations at
            # the gain margin as contradictory supervision.
            pos_w = torch.as_tensor(max(float(opportunity_pos_weight), 1.0), dtype=opp_delta_logits.dtype, device=opp_delta_logits.device)
            opp = F.binary_cross_entropy_with_logits(opp_delta_logits, labels, pos_weight=pos_w)
            terms.append(float(opportunity_weight) * opp)
        if harm_delta_logits is not None and float(harm_weight) > 0.0:
            harm_temp = float(harm_soft_label_temperature)
            harm_labels = (
                torch.sigmoid((-t_delta - float(negative_gain)) / max(harm_temp, 1.0e-4))
                if harm_temp > 0.0 else harmful_mask.to(dtype=harm_delta_logits.dtype)
            ).to(dtype=harm_delta_logits.dtype)
            harm_pos_w = torch.as_tensor(max(float(harm_pos_weight), 1.0), dtype=harm_delta_logits.dtype, device=harm_delta_logits.device)
            harm_loss = F.binary_cross_entropy_with_logits(harm_delta_logits, harm_labels, pos_weight=harm_pos_w)
            terms.append(float(harm_weight) * harm_loss)
        if opp_delta_logits is not None and float(ordinal_evidence_benefit_margin_regression_weight) > 0.0:
            # v48.37 HAF: anchor the raw-benefit factor to the same kind of
            # physically meaningful signed distance used by component-veto
            # heads. Zero logit means exactly the preregistered positive-gain
            # boundary, so P(opportunity)=0.5 has a cross-scene physical meaning
            # rather than an arbitrary class-prior calibration. The target is a
            # single continuous margin shared by every audit stratum.
            benefit_tau = max(float(ordinal_evidence_benefit_margin_temperature), 1.0e-4)
            predicted_benefit_margin = benefit_tau * opp_delta_logits
            target_benefit_margin = (
                t_delta.detach().to(dtype=predicted_benefit_margin.dtype)
                - float(positive_gain)
            )
            benefit_margin_regression = F.smooth_l1_loss(
                predicted_benefit_margin, target_benefit_margin, reduction="mean"
            )
            terms.append(
                float(ordinal_evidence_benefit_margin_regression_weight)
                * benefit_margin_regression
            )

        # v48.10 COPE ordinal evidence.  The exact-PDC advantage is
        # tri-state; train ordered benefit/non-harm logits primarily on the
        # frozen policy's top-1 candidate, with an optional weak all-candidate
        # regularizer.  This avoids the regression-to-zero failure of v48.9.
        if opp_delta_logits is not None and harm_delta_logits is not None and (
            float(ordinal_evidence_policy_top1_weight) > 0.0
            or float(ordinal_evidence_all_candidate_weight) > 0.0
        ):
            evidence_tau = max(float(certificate_policy_top1_temperature), 1.0e-4)
            benefit_target = torch.sigmoid(
                (t_delta.detach() - float(positive_gain)) / evidence_tau
            )
            harm_target = torch.sigmoid(
                (-float(negative_gain) - t_delta.detach()) / evidence_tau
            )
            gamma = max(0.0, float(ordinal_evidence_focal_gamma))

            def _focal_bce(logit: torch.Tensor, target_soft: torch.Tensor) -> torch.Tensor:
                raw = F.binary_cross_entropy_with_logits(logit, target_soft, reduction="none")
                prob = torch.sigmoid(logit)
                pt = target_soft * prob + (1.0 - target_soft) * (1.0 - prob)
                return (((1.0 - pt).clamp_min(1.0e-4) ** gamma) * raw).mean()

            if float(ordinal_evidence_all_candidate_weight) > 0.0:
                all_evidence = 0.5 * (
                    _focal_bce(opp_delta_logits, benefit_target)
                    + _focal_bce(harm_delta_logits, harm_target)
                )
                terms.append(float(ordinal_evidence_all_candidate_weight) * all_evidence)
            if float(ordinal_evidence_policy_top1_weight) > 0.0:
                policy_j = int(torch.argmax(r_delta.detach()).item())
                policy_evidence = 0.5 * (
                    _focal_bce(opp_delta_logits[policy_j:policy_j + 1], benefit_target[policy_j:policy_j + 1])
                    + _focal_bce(harm_delta_logits[policy_j:policy_j + 1], harm_target[policy_j:policy_j + 1])
                )
                terms.append(float(ordinal_evidence_policy_top1_weight) * policy_evidence)

        # v48.11 CASTER: proper three-class ordered likelihood.  The two
        # cumulative logits induce a valid simplex:
        #   p_harm = 1-P(non-harm), p_benefit=P(benefit),
        #   p_dead = P(non-harm)-P(benefit).
        # This replaces two independent BCE objectives that improved benefit AUC
        # but left harmful-vs-dead evidence nearly random.
        if opp_delta_logits is not None and harm_delta_logits is not None and (
            float(ordinal_evidence_ordered_nll_top1_weight) > 0.0
            or float(ordinal_evidence_ordered_nll_all_weight) > 0.0
            or bool(ordinal_evidence_batch_balanced)
        ):
            p_benefit = torch.sigmoid(opp_delta_logits)
            p_harm = torch.sigmoid(harm_delta_logits)
            classes = torch.ones_like(t_delta, dtype=torch.long)
            classes = torch.where(t_delta >= float(positive_gain), torch.full_like(classes, 2), classes)
            classes = torch.where(t_delta <= -float(negative_gain), torch.zeros_like(classes), classes)
            class_weights = p_benefit.new_tensor([
                float(ordinal_evidence_harm_class_weight),
                float(ordinal_evidence_dead_class_weight),
                float(ordinal_evidence_benefit_class_weight),
            ])
            benefit_loss_tail = None
            harm_loss_tail = None
            raw_benefit_binary = (classes == 2).to(dtype=p_benefit.dtype)
            benefit_binary = raw_benefit_binary
            harm_binary_bool = classes == 0
            if bool(ordinal_evidence_independent_tails):
                # v48.19 FACET-BRIDGE: independent outputs require independent
                # teacher hypotheses.  Benefit remains total PCD improvement; the
                # optional component-veto harm target fires when any safety
                # component regresses, even if total PCD still improves.
                harm_target_tail = (
                    factorized_harm_target.to(dtype=p_harm.dtype)
                    if factorized_harm_target is not None
                    else harm_binary_bool.to(dtype=p_harm.dtype)
                )
                harm_binary_bool = (
                    factorized_harm_binary
                    if factorized_harm_binary is not None
                    else harm_binary_bool
                )
                safe_admission_binary = raw_benefit_binary * (
                    ~harm_binary_bool
                ).to(dtype=raw_benefit_binary.dtype)
                # v48.21 CONCORD: the admission opportunity is not merely a raw
                # total-PCD improvement.  It is a *safe* improvement that survives
                # the component veto.  Supervising raw benefit while deployment
                # rejects benefit/harm overlap forced one logit to serve two
                # incompatible meanings and inflated opportunity on unsafe Contact
                # candidates.  Legacy checkpoints retain raw-benefit semantics.
                if bool(ordinal_evidence_safe_benefit_target):
                    benefit_binary = benefit_binary * (~harm_binary_bool).to(benefit_binary.dtype)
                benefit_loss_tail = F.binary_cross_entropy_with_logits(
                    opp_delta_logits, benefit_binary, reduction="none"
                )
                harm_loss_tail = F.binary_cross_entropy_with_logits(
                    harm_delta_logits, harm_target_tail, reduction="none"
                )
                benefit_tail_weight = torch.where(
                    benefit_binary > 0.5,
                    torch.full_like(benefit_binary, float(ordinal_evidence_benefit_class_weight)),
                    torch.full_like(benefit_binary, float(ordinal_evidence_dead_class_weight)),
                )
                harm_tail_weight = torch.where(
                    harm_binary_bool,
                    torch.full_like(harm_target_tail, float(ordinal_evidence_harm_class_weight)),
                    torch.full_like(harm_target_tail, float(ordinal_evidence_dead_class_weight)),
                )
                component_loss_tail = None
                if (
                    component_harm_delta_logits is not None
                    and factorized_component_targets is not None
                    and float(ordinal_evidence_component_tail_weight) > 0.0
                ):
                    component_loss_raw = F.binary_cross_entropy_with_logits(
                        component_harm_delta_logits,
                        factorized_component_targets.to(dtype=component_harm_delta_logits.dtype),
                        reduction="none",
                    )
                    component_binary = factorized_component_margins[:, :component_harm_delta_logits.shape[-1]] > 0.0
                    component_weight = torch.where(
                        component_binary,
                        torch.full_like(component_loss_raw, float(ordinal_evidence_harm_class_weight)),
                        torch.full_like(component_loss_raw, float(ordinal_evidence_dead_class_weight)),
                    )
                    raw_reliability = ordinal_evidence_component_reliability
                    if raw_reliability is None:
                        reliability_values = []
                    elif isinstance(raw_reliability, str):
                        reliability_text = raw_reliability.strip()
                        if reliability_text.lower() in {"", "none", "null", "~"}:
                            reliability_values = []
                        else:
                            reliability_values = [
                                float(x.strip()) for x in reliability_text.split(",") if x.strip()
                            ]
                    else:
                        reliability_values = [float(x) for x in raw_reliability]
                    if not reliability_values:
                        reliability_values = [1.0] * component_harm_delta_logits.shape[-1]
                    if len(reliability_values) < component_harm_delta_logits.shape[-1]:
                        reliability_values.extend(
                            [1.0] * (component_harm_delta_logits.shape[-1] - len(reliability_values))
                        )
                    component_reliability = component_loss_raw.new_tensor(
                        [
                            min(1.0, max(0.0, x))
                            for x in reliability_values[: component_harm_delta_logits.shape[-1]]
                        ]
                    )
                    weighted_component_loss = (
                        component_loss_raw * component_weight * component_reliability
                    )
                    component_loss_tail = weighted_component_loss.sum(dim=-1) / (
                        component_reliability.sum().clamp_min(1.0e-6)
                    )
                nll = 0.5 * (
                    benefit_tail_weight * benefit_loss_tail
                    + harm_tail_weight * harm_loss_tail
                )
                if component_loss_tail is not None:
                    nll = nll + float(ordinal_evidence_component_tail_weight) * component_loss_tail
                if (
                    component_harm_delta_logits is not None
                    and factorized_component_margins is not None
                    and float(ordinal_evidence_component_margin_regression_weight) > 0.0
                ):
                    # v48.30: BCE identifies the side of each veto boundary but
                    # discards distance-to-boundary.  Regressing the signed margins
                    # makes the factor heads usable as a continuous safety-slack
                    # projection shared by Safe, Near and Contact.
                    predicted_component_margins = (
                        float(ordinal_evidence_factorized_harm_temperature)
                        * component_harm_delta_logits
                    )
                    target_component_margins_raw = factorized_component_margins[
                        :, : component_harm_delta_logits.shape[-1]
                    ].to(dtype=predicted_component_margins.dtype)
                    target_component_margins_regression = component_margin_regression_targets(
                        target_component_margins_raw,
                        mode=ordinal_evidence_component_margin_target_mode,
                        target_scale=ordinal_evidence_component_margin_target_scale,
                        canonical_scales=ordinal_evidence_component_margin_canonical_scales,
                    )
                    regression_raw = F.smooth_l1_loss(
                        predicted_component_margins,
                        target_component_margins_regression,
                        reduction="none",
                    )
                    ncomp = int(component_harm_delta_logits.shape[-1])
                    component_reliability = predicted_component_margins.new_tensor(
                        _component_vector(
                            ordinal_evidence_component_reliability,
                            ncomp=ncomp, default=1.0, clamp_unit=True,
                        )
                    )
                    # v48.55 separates sign support from explicit continuous-
                    # distance regression support. DRS can remain fully
                    # supervised by component BCE while being excluded from
                    # SmoothL1 magnitude regression in the X factor.
                    regression_reliability = predicted_component_margins.new_tensor(
                        _component_vector(
                            ordinal_evidence_component_margin_regression_reliability,
                            ncomp=ncomp, default=1.0, clamp_unit=True,
                        )
                    )
                    effective_regression_reliability = component_reliability * regression_reliability
                    regression = (regression_raw * effective_regression_reliability).sum() / (
                        predicted_component_margins.shape[0]
                        * effective_regression_reliability.sum().clamp_min(1.0e-6)
                    )
                    terms.append(
                        float(ordinal_evidence_component_margin_regression_weight)
                        * regression
                    )
                if (
                    component_harm_delta_logits is not None
                    and factorized_component_margins is not None
                    and (
                        float(ordinal_evidence_component_underestimation_weight) > 0.0
                        or float(ordinal_evidence_safe_positive_component_overestimation_weight) > 0.0
                        or float(ordinal_evidence_joint_reserve_regression_weight) > 0.0
                    )
                ):
                    # v48.38 RFR: the certificate failure is concentrated in
                    # the low-predicted-harm tail, where some truly large veto
                    # violations are underestimated while safe-positive actions
                    # are overestimated.  Keep dense signed-margin supervision,
                    # but add an asymmetric per-component penalty for dangerous
                    # underestimation and a joint physical reserve target.
                    predicted_component_margins = (
                        float(ordinal_evidence_factorized_harm_temperature)
                        * component_harm_delta_logits
                    )
                    target_component_margins = factorized_component_margins[
                        :, : component_harm_delta_logits.shape[-1]
                    ].to(dtype=predicted_component_margins.dtype)
                    raw_reliability = ordinal_evidence_component_reliability
                    if raw_reliability is None:
                        reliability_values = []
                    elif isinstance(raw_reliability, str):
                        reliability_text = raw_reliability.strip()
                        if reliability_text.lower() in {"", "none", "null", "~"}:
                            reliability_values = []
                        else:
                            reliability_values = [
                                float(x.strip()) for x in reliability_text.split(",") if x.strip()
                            ]
                    else:
                        reliability_values = [float(x) for x in raw_reliability]
                    if not reliability_values:
                        reliability_values = [1.0] * component_harm_delta_logits.shape[-1]
                    if len(reliability_values) < component_harm_delta_logits.shape[-1]:
                        reliability_values.extend(
                            [1.0] * (component_harm_delta_logits.shape[-1] - len(reliability_values))
                        )
                    component_reliability = predicted_component_margins.new_tensor(
                        [
                            min(1.0, max(0.0, x))
                            for x in reliability_values[: component_harm_delta_logits.shape[-1]]
                        ]
                    )
                    supported = component_reliability > 0.0
                    if not bool(supported.any()):
                        raise RuntimeError("RFR joint reserve requires at least one supported component")

                    target_benefit_margin_for_tail = (
                        t_delta.detach().to(dtype=predicted_component_margins.dtype)
                        - float(positive_gain)
                    )
                    if float(ordinal_evidence_component_underestimation_weight) > 0.0:
                        # Only true veto violations receive the extra one-sided
                        # false-safe penalty. This directly targets the observed
                        # certificate tail (large positive teacher margin but low
                        # predicted harm) without globally biasing safe examples.
                        harmful_coordinate = (target_component_margins > 0.0).to(
                            dtype=predicted_component_margins.dtype
                        )
                        under = torch.relu(
                            target_component_margins - predicted_component_margins
                        ) * harmful_coordinate * component_reliability
                        denom = (
                            harmful_coordinate * component_reliability
                        ).sum().clamp_min(1.0)
                        terms.append(
                            float(ordinal_evidence_component_underestimation_weight)
                            * under.sum() / denom
                        )

                    if float(ordinal_evidence_safe_positive_component_overestimation_weight) > 0.0:
                        # The opposite error dominates missed opportunities: truly
                        # safe-beneficial candidates are often assigned very high
                        # predicted harm. Penalise that direction only on the same
                        # continuous safe-positive definition used by the gate.
                        # "Safe-positive" must use the complete teacher veto
                        # definition, not only the subset of learned coordinates
                        # with non-zero reliability.  Otherwise a row that is
                        # harmful on an unsupported coordinate could incorrectly
                        # receive a loss that pushes the supported harm estimates
                        # downward.  Reliability still controls which predicted
                        # coordinates receive gradients.
                        target_full_worst_margin_for_tail = target_component_margins.amax(dim=-1)
                        safe_positive_row = (
                            (target_benefit_margin_for_tail > 0.0)
                            & (target_full_worst_margin_for_tail <= 0.0)
                        ).to(dtype=predicted_component_margins.dtype)
                        over = torch.relu(
                            predicted_component_margins - target_component_margins
                        ) * component_reliability * safe_positive_row.unsqueeze(-1)
                        denom = (
                            component_reliability.sum() * safe_positive_row.sum()
                        ).clamp_min(1.0)
                        terms.append(
                            float(ordinal_evidence_safe_positive_component_overestimation_weight)
                            * over.sum() / denom
                        )

                    if (
                        opp_delta_logits is not None
                        and float(ordinal_evidence_joint_reserve_regression_weight) > 0.0
                    ):
                        benefit_tau = max(float(ordinal_evidence_benefit_margin_temperature), 1.0e-4)
                        predicted_benefit_margin = benefit_tau * opp_delta_logits
                        target_benefit_margin = (
                            t_delta.detach().to(dtype=predicted_benefit_margin.dtype)
                            - float(positive_gain)
                        )
                        # Unsupported learned components cannot define the
                        # differentiable reserve. Independent measured hard vetoes
                        # remain active downstream exactly as before.
                        large_negative = predicted_component_margins.new_tensor(-1.0e6)
                        pred_supported = torch.where(
                            supported.unsqueeze(0), predicted_component_margins, large_negative
                        )
                        target_supported = torch.where(
                            supported.unsqueeze(0), target_component_margins, large_negative
                        )
                        predicted_safety_headroom = -pred_supported.amax(dim=-1)
                        target_safety_headroom = -target_supported.amax(dim=-1)
                        predicted_joint_reserve = torch.minimum(
                            predicted_benefit_margin, predicted_safety_headroom
                        )
                        target_joint_reserve = torch.minimum(
                            target_benefit_margin, target_safety_headroom
                        ).detach()
                        reserve_raw = F.smooth_l1_loss(
                            predicted_joint_reserve, target_joint_reserve, reduction="none"
                        )
                        boundary_width = max(
                            1.0e-6, float(ordinal_evidence_joint_reserve_boundary_width)
                        )
                        boundary_weight = max(
                            0.0, float(ordinal_evidence_joint_reserve_boundary_weight)
                        )
                        # The deployment decision changes at reserve=0; preserve
                        # dense population training but spend extra capacity close
                        # to that shared physical boundary. No regime/bucket label
                        # enters this weighting.
                        near_boundary = (target_joint_reserve.abs() <= boundary_width).to(
                            dtype=reserve_raw.dtype
                        )
                        reserve_weights = 1.0 + boundary_weight * near_boundary
                        reserve_loss = (reserve_raw * reserve_weights).sum() / (
                            reserve_weights.sum().clamp_min(1.0e-6)
                        )
                        terms.append(
                            float(ordinal_evidence_joint_reserve_regression_weight)
                            * reserve_loss
                        )

                if admission_delta_logits is not None and float(ordinal_evidence_admission_weight) > 0.0:
                    # v48.22 COVENANT: direct safe-admission supervision is a
                    # third hypothesis, distinct from raw benefit and component
                    # harm.  False-safe harmful candidates receive extra weight
                    # because they dominate certificate precision/UCB failures.
                    admission_raw_loss = F.binary_cross_entropy_with_logits(
                        admission_delta_logits, safe_admission_binary, reduction="none"
                    )
                    admission_weights = torch.where(
                        safe_admission_binary > 0.5,
                        torch.full_like(admission_raw_loss, float(ordinal_evidence_admission_pos_weight)),
                        torch.ones_like(admission_raw_loss),
                    )
                    admission_weights = torch.where(
                        harm_binary_bool,
                        admission_weights * float(ordinal_evidence_admission_harm_negative_weight),
                        admission_weights,
                    )
                    terms.append(
                        float(ordinal_evidence_admission_weight)
                        * (admission_raw_loss * admission_weights).mean()
                    )
            else:
                p_dead = (1.0 - p_benefit - p_harm).clamp_min(1.0e-6)
                probs = torch.stack([p_harm, p_dead, p_benefit], dim=-1).clamp_min(1.0e-6)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
                nll = -torch.log(probs[torch.arange(probs.shape[0], device=probs.device), classes])
                nll = nll * class_weights[classes]
            # v48.14 PRISM: calibration-domain evidence adaptation must focus on
            # the errors that invalidate a selective safety certificate.  Static
            # class weights alone still let the adapter minimise loss by fitting
            # abundant dead-zone samples.  Dynamically upweight false-safe
            # harmful proposals and missed beneficial proposals, while detaching
            # the hardness factor so the weighting cannot be gamed by the model.
            hard_gamma = max(0.0, float(ordinal_evidence_hard_example_gamma))
            if float(ordinal_evidence_hard_harm_weight) > 0.0:
                harm_hardness = (1.0 - p_harm.detach()).clamp(0.0, 1.0).pow(hard_gamma)
                nll = torch.where(
                    harm_binary_bool if bool(ordinal_evidence_independent_tails) else classes == 0,
                    nll * (1.0 + float(ordinal_evidence_hard_harm_weight) * harm_hardness),
                    nll,
                )
            if float(ordinal_evidence_hard_benefit_weight) > 0.0:
                benefit_hardness = (1.0 - p_benefit.detach()).clamp(0.0, 1.0).pow(hard_gamma)
                nll = torch.where(
                    benefit_binary > 0.5 if bool(ordinal_evidence_independent_tails) else classes == 2,
                    nll * (1.0 + float(ordinal_evidence_hard_benefit_weight) * benefit_hardness),
                    nll,
                )
            replace_evidence_erm = bool(
                ordinal_evidence_batch_balanced and ordinal_evidence_balanced_replaces_erm
            )
            if float(ordinal_evidence_ordered_nll_all_weight) > 0.0 and not replace_evidence_erm:
                terms.append(float(ordinal_evidence_ordered_nll_all_weight) * nll.mean())
            if float(ordinal_evidence_ordered_nll_top1_weight) > 0.0 and not replace_evidence_erm:
                policy_j = int(torch.argmax(r_delta.detach()).item())
                terms.append(float(ordinal_evidence_ordered_nll_top1_weight) * nll[policy_j])
            else:
                policy_j = int(torch.argmax(r_delta.detach()).item())

            # v48.13 TERRA: train evidence on every member of the frozen top-k
            # proposal, matching deployment and exposing hard runner-up errors.
            proposal_k = min(max(1, int(ordinal_evidence_proposal_topk)), int(r_delta.numel()))
            proposal_idx = torch.topk(r_delta.detach(), k=proposal_k).indices
            if float(ordinal_evidence_proposal_topk_weight) > 0.0 and not replace_evidence_erm:
                decay = max(0.0, min(1.0, float(ordinal_evidence_proposal_rank_decay)))
                rank_weights = nll.new_tensor([decay ** i for i in range(proposal_k)])
                rank_weights = rank_weights / rank_weights.sum().clamp_min(1.0e-6)
                terms.append(float(ordinal_evidence_proposal_topk_weight) * (rank_weights * nll[proposal_idx]).sum())

            # v48.16 ANCHOR: adaptation sets are dead-zone heavy. Average
            # evidence loss per observed class before averaging classes, so the
            # tiny adapter cannot win by predicting abstention for every proposal.
            proposal_classes = classes[proposal_idx]
            proposal_benefit_mask = benefit_binary[proposal_idx] > 0.5
            proposal_harm_mask = harm_binary_bool[proposal_idx]
            if bool(ordinal_evidence_batch_balanced):
                regime_id = int(bid[nom].detach().item())
                for local_idx in proposal_idx:
                    if bool(ordinal_evidence_factorized_harm) and benefit_loss_tail is not None and harm_loss_tail is not None:
                        evidence_proposal_records.append((
                            regime_id,
                            benefit_loss_tail[local_idx],
                            harm_loss_tail[local_idx],
                            p_benefit[local_idx],
                            p_harm[local_idx],
                            bool(benefit_binary[local_idx].detach().item() > 0.5),
                            bool(harm_binary_bool[local_idx].detach().item()),
                        ))
                    else:
                        evidence_proposal_records.append((
                            regime_id,
                            nll[local_idx],
                            p_benefit[local_idx],
                            p_harm[local_idx],
                            int(classes[local_idx].detach().item()),
                        ))
            elif float(ordinal_evidence_class_balanced_weight) > 0.0:
                class_terms = []
                if bool(ordinal_evidence_independent_tails):
                    for tail_loss, tail_mask in (
                        (benefit_loss_tail[proposal_idx], proposal_benefit_mask),
                        (harm_loss_tail[proposal_idx], proposal_harm_mask),
                    ):
                        for label in (False, True):
                            mask = tail_mask == label
                            if bool(mask.any()):
                                class_terms.append(tail_loss[mask].mean())
                else:
                    for class_id in (0, 1, 2):
                        mask = proposal_classes == class_id
                        if bool(mask.any()):
                            class_terms.append(nll[proposal_idx][mask].mean())
                if class_terms:
                    terms.append(float(ordinal_evidence_class_balanced_weight) * torch.stack(class_terms).mean())

            target_prob = min(0.99, max(0.51, float(ordinal_evidence_target_probability)))
            if not bool(ordinal_evidence_batch_balanced):
                if float(ordinal_evidence_benefit_margin_weight) > 0.0:
                    benefit_idx = proposal_idx[proposal_benefit_mask]
                    if benefit_idx.numel():
                        terms.append(float(ordinal_evidence_benefit_margin_weight) * F.relu(target_prob - p_benefit[benefit_idx]).mean())
                if float(ordinal_evidence_harm_margin_weight) > 0.0:
                    harm_idx_margin = proposal_idx[proposal_harm_mask]
                    if harm_idx_margin.numel():
                        terms.append(float(ordinal_evidence_harm_margin_weight) * F.relu(target_prob - p_harm[harm_idx_margin]).mean())

            # Same-group counterfactual comparisons cancel shared scene severity.
            intra_margin = float(ordinal_evidence_intragroup_margin)
            if float(ordinal_evidence_intragroup_benefit_weight) > 0.0:
                pos_idx = proposal_idx[proposal_benefit_mask]
                neg_idx = proposal_idx[~proposal_benefit_mask]
                if pos_idx.numel() and neg_idx.numel():
                    benefit_pairs = opp_delta_logits[pos_idx].unsqueeze(1) - opp_delta_logits[neg_idx].unsqueeze(0)
                    terms.append(float(ordinal_evidence_intragroup_benefit_weight) * F.softplus(intra_margin - benefit_pairs).mean())
            if float(ordinal_evidence_intragroup_harm_weight) > 0.0:
                harm_idx = proposal_idx[proposal_harm_mask]
                safe_idx = proposal_idx[~proposal_harm_mask]
                if harm_idx.numel() and safe_idx.numel():
                    harm_pairs = harm_delta_logits[harm_idx].unsqueeze(1) - harm_delta_logits[safe_idx].unsqueeze(0)
                    terms.append(float(ordinal_evidence_intragroup_harm_weight) * F.softplus(intra_margin - harm_pairs).mean())
            if (
                float(ordinal_evidence_pairwise_benefit_weight) > 0.0
                or float(ordinal_evidence_pairwise_harm_weight) > 0.0
            ):
                policy_class = int(classes[policy_j].detach().item())
                evidence_policy_records.append((
                    int(bid[nom].detach().item()),
                    opp_delta_logits[policy_j],
                    harm_delta_logits[policy_j],
                    policy_class,
                ))

        # v48.4 DRA-RCD legacy admission distribution.  Older objectives and
        # checkpoints retain this path unchanged.
        recovery_admission_logits = p_delta
        if opp_delta_logits is not None:
            recovery_admission_logits = recovery_admission_logits + float(opportunity_admission_weight) * F.logsigmoid(opp_delta_logits)
        if harm_delta_logits is not None:
            recovery_admission_logits = recovery_admission_logits + float(harm_admission_weight) * F.logsigmoid(-harm_delta_logits)
        admission_class_logits = torch.cat([score[nom:nom + 1] * 0.0, recovery_admission_logits], dim=0) / tau
        rank_class_logits = torch.cat([rank_score[nom:nom + 1] * 0.0, r_delta], dim=0) / tau
        admission_harm_mask = (
            factorized_harm_binary
            if factorized_harm_binary is not None
            else harmful_mask
        )
        safe_positive_mask = pos_mask & (~admission_harm_mask)
        target_class = 0
        if bool(safe_positive_mask.any()):
            safe_indices = torch.where(safe_positive_mask)[0]
            safe_best = safe_indices[torch.argmax(t_delta[safe_indices])]
            target_class = 1 + int(safe_best.item())
        target_tensor = torch.tensor([target_class], dtype=torch.long, device=score.device)

        # v48.20 UNISON deployment-exact safe-set admission.  Calibration and
        # closed loop first freeze a rank-based top-k proposal, then rerank only
        # that proposal by sigmoid(benefit)-sigmoid(harm).  v48.19 trained a
        # different all-candidate score (frozen PCD delta plus log-sigmoid tails),
        # so even a well-fitted candidate classifier was not optimized for the
        # action actually deployed.  Restrict both the target set and the loss to
        # the frozen top-k and use the exact deployed evidence score.  No bucket or
        # regime identifier enters this path.
        safe_set_logits = admission_class_logits
        safe_set_harm_mask = admission_harm_mask
        safe_set_positive_mask = safe_positive_mask
        safe_set_teacher_delta = t_delta
        unison_safe_set = bool(
            ordinal_evidence_independent_tails
            and ordinal_evidence_factorized_harm
            and opp_delta_logits is not None
            and harm_delta_logits is not None
        )
        if unison_safe_set:
            deployment_k = min(max(1, int(ordinal_evidence_proposal_topk)), int(r_delta.numel()))
            deployment_idx = torch.topk(r_delta.detach(), k=deployment_k).indices
            if admission_delta_logits is not None:
                deployment_score = torch.sigmoid(admission_delta_logits[deployment_idx]) - 0.5
            else:
                deployment_score = (
                    torch.sigmoid(opp_delta_logits[deployment_idx])
                    * (1.0 - torch.sigmoid(harm_delta_logits[deployment_idx]))
                ) - 0.5
            safe_set_logits = torch.cat(
                [deployment_score.new_zeros((1,)), deployment_score], dim=0
            ) / tau
            safe_set_harm_mask = admission_harm_mask[deployment_idx]
            safe_set_positive_mask = pos_mask[deployment_idx] & (~safe_set_harm_mask)
            safe_set_teacher_delta = t_delta[deployment_idx]

            # v48.23 FRONTIER: distil continuous raw-gain ordering inside the
            # exact frozen proposal. Binary benefit AUC alone cannot identify the
            # best post-contact action or reduce top-1 regret.
            if float(ordinal_evidence_benefit_listwise_weight) > 0.0:
                benefit_tau = max(float(ordinal_evidence_benefit_listwise_temperature), 1.0e-3)
                student_benefit_logits = torch.cat([
                    opp_delta_logits.new_zeros((1,)),
                    opp_delta_logits[deployment_idx],
                ]) / benefit_tau
                teacher_benefit_logits = torch.cat([
                    safe_set_teacher_delta.new_zeros((1,)),
                    safe_set_teacher_delta,
                ]) / benefit_tau
                teacher_benefit_prob = torch.softmax(teacher_benefit_logits, dim=0).detach()
                terms.append(
                    float(ordinal_evidence_benefit_listwise_weight)
                    * F.kl_div(
                        torch.log_softmax(student_benefit_logits, dim=0),
                        teacher_benefit_prob, reduction="sum",
                    )
                )

            # v48.24 SUPPORT-BRIDGE: regress and rank the exact deployed
            # candidate-vs-nominal admission score against a continuous
            # *safe utility*.  Raw-benefit listwise supervision in v48.23 could
            # rank a beneficial-but-harmful action above nominal and then rely on
            # a separate sparse veto.  The certificate, however, thresholds the
            # final admission score.  Harmful candidates therefore receive a
            # strictly negative target while non-harmful candidates retain their
            # continuous PCD advantage.  This makes score correlation, top-1
            # regret and safety-frontier separation share one deployment-exact
            # target rather than three weakly coupled objectives.
            if admission_delta_logits is not None:
                harmful_floor = safe_set_teacher_delta.new_full(
                    safe_set_teacher_delta.shape, max(float(positive_gain), 1.0e-3)
                )
                # Runtime deploys sigmoid(admission_delta)-0.5. Train the exact
                # same score and range. The v48.24/v48.25 tanh(logit/2) proxy was
                # exactly two times larger and broke the train/certificate scale.
                safe_utility_target = torch.where(
                    safe_set_harm_mask,
                    -torch.maximum(safe_set_teacher_delta.abs(), harmful_floor),
                    safe_set_teacher_delta,
                ).clamp(-0.5, 0.5)
                deployed_safe_utility = (
                    torch.sigmoid(admission_delta_logits[deployment_idx]) - 0.5
                )
                if float(ordinal_evidence_safe_utility_regression_weight) > 0.0:
                    terms.append(
                        float(ordinal_evidence_safe_utility_regression_weight)
                        * F.smooth_l1_loss(
                            deployed_safe_utility, safe_utility_target.detach()
                        )
                    )
                if float(ordinal_evidence_safe_utility_listwise_weight) > 0.0:
                    safe_tau = max(
                        float(ordinal_evidence_safe_utility_temperature), 1.0e-3
                    )
                    # Match the exact deployed selector score. Using raw logits
                    # here while the teacher lived in [-0.5, 0.5] made the
                    # listwise gradient dominate regression and explains the
                    # v48.26 C/D degradation.
                    safe_student = torch.cat([
                        deployed_safe_utility.new_zeros((1,)),
                        deployed_safe_utility,
                    ]) / safe_tau
                    safe_teacher = torch.cat([
                        safe_utility_target.new_zeros((1,)),
                        safe_utility_target,
                    ]) / safe_tau
                    safe_teacher_prob = torch.softmax(safe_teacher, dim=0).detach()
                    terms.append(
                        float(ordinal_evidence_safe_utility_listwise_weight)
                        * F.kl_div(
                            torch.log_softmax(safe_student, dim=0),
                            safe_teacher_prob, reduction="sum",
                        )
                    )

                # v48.33 ELIGIBLE-SET POLICY: calibration and runtime first
                # filter the frozen proposal by opportunity/harm, then rerank
                # the remaining candidates by admission evidence.  Earlier
                # listwise losses optimized evidence independently of this
                # filter, so a good runner-up received no deployment gradient
                # when an ineligible high-evidence candidate occupied top-1.
                # This differentiable categorical policy uses the same order:
                # proposal -> soft eligibility -> evidence -> one action/nominal.
                # It is shared by every regime and contains no regime ID.
                if (
                    float(ordinal_evidence_eligible_policy_weight) > 0.0
                    and opp_delta_logits is not None
                    and harm_delta_logits is not None
                ):
                    policy_tau = max(
                        float(ordinal_evidence_eligible_policy_temperature), 1.0e-3
                    )
                    gate_tau = max(
                        float(ordinal_evidence_eligibility_logit_temperature), 1.0e-3
                    )
                    opp_threshold = min(max(
                        float(ordinal_evidence_eligible_opportunity_threshold), 1.0e-4
                    ), 1.0 - 1.0e-4)
                    harm_threshold = min(max(
                        float(ordinal_evidence_eligible_harm_threshold), 1.0e-4
                    ), 1.0 - 1.0e-4)
                    opp_threshold_logit = deployed_safe_utility.new_tensor(
                        math.log(opp_threshold / (1.0 - opp_threshold))
                    )
                    harm_threshold_logit = deployed_safe_utility.new_tensor(
                        math.log(harm_threshold / (1.0 - harm_threshold))
                    )
                    proposal_opp_logits = opp_delta_logits[deployment_idx]
                    proposal_harm_logits = harm_delta_logits[deployment_idx]
                    log_soft_eligibility = (
                        F.logsigmoid(
                            (proposal_opp_logits - opp_threshold_logit) / gate_tau
                        )
                        + F.logsigmoid(
                            (harm_threshold_logit - proposal_harm_logits) / gate_tau
                        )
                    )
                    eligible_student = torch.cat([
                        deployed_safe_utility.new_zeros((1,)),
                        deployed_safe_utility / policy_tau + log_soft_eligibility,
                    ])
                    eligible_teacher = torch.cat([
                        safe_utility_target.new_zeros((1,)),
                        safe_utility_target / policy_tau,
                    ])
                    eligible_teacher_prob = torch.softmax(
                        eligible_teacher, dim=0
                    ).detach()
                    terms.append(
                        float(ordinal_evidence_eligible_policy_weight)
                        * F.kl_div(
                            torch.log_softmax(eligible_student, dim=0),
                            eligible_teacher_prob, reduction="sum",
                        )
                    )

                # v48.34 HARD-BOUNDARY CONTINUATION: v48.33 increased soft
                # eligible-set mass without moving candidates across the exact
                # opportunity/harm/admission boundaries used by checkpointing and
                # deployment.  Apply candidate-level margin constraints inside the
                # same frozen proposal.  Safe-beneficial actions are pulled into the
                # executable envelope; component-harmful actions are pushed through
                # the harm boundary and below nominal admission.  This is global,
                # continuous and regime-agnostic.
                if (
                    float(ordinal_evidence_eligibility_boundary_weight) > 0.0
                    and opp_delta_logits is not None
                    and harm_delta_logits is not None
                    and admission_delta_logits is not None
                ):
                    boundary_weight = float(ordinal_evidence_eligibility_boundary_weight)
                    boundary_margin = max(0.0, float(ordinal_evidence_eligibility_boundary_margin))
                    opp_threshold = min(max(
                        float(ordinal_evidence_eligible_opportunity_threshold), 1.0e-4
                    ), 1.0 - 1.0e-4)
                    harm_threshold = min(max(
                        float(ordinal_evidence_eligible_harm_threshold), 1.0e-4
                    ), 1.0 - 1.0e-4)
                    opp_threshold_logit = deployed_safe_utility.new_tensor(
                        math.log(opp_threshold / (1.0 - opp_threshold))
                    )
                    harm_threshold_logit = deployed_safe_utility.new_tensor(
                        math.log(harm_threshold / (1.0 - harm_threshold))
                    )
                    proposal_opp_logits = opp_delta_logits[deployment_idx]
                    proposal_harm_logits = harm_delta_logits[deployment_idx]
                    proposal_admission_logits = admission_delta_logits[deployment_idx]
                    boundary_terms: list[torch.Tensor] = []
                    if bool(safe_set_positive_mask.any()):
                        safe_opp = proposal_opp_logits[safe_set_positive_mask]
                        safe_harm = proposal_harm_logits[safe_set_positive_mask]
                        safe_admission = proposal_admission_logits[safe_set_positive_mask]
                        boundary_terms.extend([
                            F.softplus(opp_threshold_logit + boundary_margin - safe_opp).mean(),
                            F.softplus(safe_harm - (harm_threshold_logit - boundary_margin)).mean(),
                            F.softplus(boundary_margin - safe_admission).mean(),
                        ])
                    if bool(safe_set_harm_mask.any()):
                        unsafe_harm = proposal_harm_logits[safe_set_harm_mask]
                        unsafe_admission = proposal_admission_logits[safe_set_harm_mask]
                        boundary_terms.extend([
                            F.softplus(harm_threshold_logit + boundary_margin - unsafe_harm).mean(),
                            F.softplus(unsafe_admission + boundary_margin).mean(),
                        ])
                    dead_mask = (~safe_set_positive_mask) & (~safe_set_harm_mask)
                    if bool(dead_mask.any()):
                        dead_opp = proposal_opp_logits[dead_mask]
                        dead_admission = proposal_admission_logits[dead_mask]
                        boundary_terms.extend([
                            0.5 * F.softplus(dead_opp - (opp_threshold_logit - boundary_margin)).mean(),
                            0.5 * F.softplus(dead_admission + boundary_margin).mean(),
                        ])
                    if boundary_terms:
                        terms.append(boundary_weight * torch.stack(boundary_terms).mean())

            # v48.29 VETO-RANK: sparse safe-positive groups need a direct
            # execution-aligned margin.  The teacher-best safe action must beat
            # nominal and the single hardest non-safe proposal member.  On
            # groups with no safe opportunity every recovery score is pushed
            # below nominal.  This focuses gradient on the actual top-1 failure
            # rather than averaging over many easy candidates.
            if (
                float(ordinal_evidence_safe_hard_negative_weight) > 0.0
                and admission_delta_logits is not None
                and deployed_safe_utility.numel()
            ):
                margin = float(ordinal_evidence_safe_hard_negative_margin)
                teacher_scale = max(
                    0.0, float(ordinal_evidence_safe_hard_negative_teacher_scale)
                )
                safe_idx = torch.where(safe_set_positive_mask)[0]
                if safe_idx.numel():
                    best_safe = safe_idx[torch.argmax(safe_set_teacher_delta[safe_idx])]
                    negative_mask = ~safe_set_positive_mask
                    negative_scores = deployed_safe_utility[negative_mask]
                    hard_negative = torch.cat([
                        deployed_safe_utility.new_zeros((1,)), negative_scores
                    ]).max()
                    negative_teacher = torch.cat([
                        safe_utility_target.new_zeros((1,)),
                        safe_utility_target[negative_mask],
                    ]).max()
                    adaptive_teacher_gap = (
                        safe_utility_target[best_safe] - negative_teacher
                    ).clamp(min=0.0, max=0.25)
                    required_margin = margin + teacher_scale * adaptive_teacher_gap
                    safe_gap = deployed_safe_utility[best_safe] - hard_negative
                    terms.append(
                        float(ordinal_evidence_safe_hard_negative_weight)
                        * F.softplus(required_margin - safe_gap)
                    )
                else:
                    max_recovery = deployed_safe_utility.max()
                    teacher_noop_depth = (-safe_utility_target.max()).clamp(
                        min=0.0, max=0.25
                    )
                    required_margin = margin + teacher_scale * teacher_noop_depth
                    terms.append(
                        float(ordinal_evidence_safe_hard_negative_weight)
                        * F.softplus(required_margin + max_recovery)
                    )

            # Directly train the high-benefit safety frontier rather than global
            # harmful-vs-dead discrimination. Safe beneficial candidates must
            # outrank beneficial-but-component-harmful candidates.
            if (
                float(ordinal_evidence_frontier_pairwise_weight) > 0.0
                and admission_delta_logits is not None
            ):
                frontier_safe = torch.where(safe_set_positive_mask)[0]
                frontier_bad = torch.where(pos_mask[deployment_idx] & safe_set_harm_mask)[0]
                if frontier_safe.numel() and frontier_bad.numel():
                    # Frontier margin is expressed in deployed safe-utility
                    # units, not unconstrained logit units.
                    safe_logits = deployed_safe_utility[frontier_safe]
                    bad_logits = deployed_safe_utility[frontier_bad]
                    frontier_pairs = safe_logits.unsqueeze(1) - bad_logits.unsqueeze(0)
                    terms.append(
                        float(ordinal_evidence_frontier_pairwise_weight)
                        * F.softplus(
                            float(ordinal_evidence_frontier_pairwise_margin) - frontier_pairs
                        ).mean()
                    )

        # Multiple-instance safe-opportunity objective.  The deployed decision
        # first asks whether the frozen top-k contains *any* safe beneficial
        # recovery.  Candidate BCE and listwise ranking alone do not directly
        # supervise this group event.  Noisy-OR is permutation invariant, shares
        # one model across all regimes, and gives useful gradients even when the
        # exact best candidate is ambiguous.
        if (
            float(ordinal_evidence_group_opportunity_weight) > 0.0
            and unison_safe_set
            and opp_delta_logits is not None
        ):
            if admission_delta_logits is not None:
                deployment_prob = torch.sigmoid(admission_delta_logits[deployment_idx])
            else:
                deployment_prob = (
                    torch.sigmoid(opp_delta_logits[deployment_idx])
                    * (1.0 - torch.sigmoid(harm_delta_logits[deployment_idx]))
                )
            deployment_prob = deployment_prob.clamp(1.0e-6, 1.0 - 1.0e-6)
            if bool(ordinal_evidence_categorical_group_policy):
                # One mutually exclusive action or nominal is executed. Noisy-OR
                # incorrectly treats top-k candidates as independent events.
                group_policy_prob = torch.softmax(safe_set_logits, dim=0)
                any_safe_prob = group_policy_prob[1:].sum().clamp(1.0e-6, 1.0 - 1.0e-6)
            else:
                any_safe_prob = 1.0 - torch.prod(1.0 - deployment_prob)
            any_safe_target = safe_set_positive_mask.any().to(dtype=any_safe_prob.dtype)
            group_opportunity_loss = F.binary_cross_entropy(
                any_safe_prob.clamp(1.0e-6, 1.0 - 1.0e-6), any_safe_target
            )
            terms.append(float(ordinal_evidence_group_opportunity_weight) * group_opportunity_loss)

        safe_set_target_class = 0
        if bool(safe_set_positive_mask.any()):
            safe_indices = torch.where(safe_set_positive_mask)[0]
            safe_best = safe_indices[torch.argmax(safe_set_teacher_delta[safe_indices])]
            safe_set_target_class = 1 + int(safe_best.item())
        safe_set_target_tensor = torch.tensor(
            [safe_set_target_class], dtype=torch.long, device=score.device
        )
        if float(setwise_admission_weight) > 0.0:
            # Positive-but-harmful overlap candidates are never admission targets;
            # they remain benefit positives for the independent benefit tail and
            # are rejected by the non-compensatory component-veto tail.
            if bool(safe_set_positive_mask.any()):
                teacher_logits = torch.full_like(safe_set_logits, -30.0)
                safe_indices = torch.where(safe_set_positive_mask)[0]
                set_tau = max(float(ordinal_evidence_safe_set_temperature), 1.0e-3)
                teacher_logits[1 + safe_indices] = safe_set_teacher_delta[safe_indices] / set_tau
                teacher_prob = torch.softmax(teacher_logits, dim=0).detach()
                admission_log_prob = torch.log_softmax(safe_set_logits, dim=0)
                set_loss = F.kl_div(admission_log_prob, teacher_prob, reduction="sum")
            else:
                set_loss = F.cross_entropy(safe_set_logits.unsqueeze(0), safe_set_target_tensor)
            terms.append(float(setwise_admission_weight) * set_loss)

        if float(selective_risk_weight) > 0.0 or float(selective_coverage_weight) > 0.0:
            policy_prob = torch.softmax(safe_set_logits, dim=0)
            recovery_prob = policy_prob[1:]
            harmful_mass = (recovery_prob * safe_set_harm_mask.to(recovery_prob.dtype)).sum()
            risk_excess = F.relu(harmful_mass - float(selective_harm_budget))
            if float(selective_risk_weight) > 0.0:
                terms.append(float(selective_risk_weight) * risk_excess.square())
            if float(selective_coverage_weight) > 0.0 and bool(safe_set_positive_mask.any()):
                positive_mass = (recovery_prob * safe_set_positive_mask.to(recovery_prob.dtype)).sum()
                coverage_shortfall = F.relu(float(selective_coverage_target) - positive_mass)
                terms.append(float(selective_coverage_weight) * coverage_shortfall.square())

        if float(policy_distill_weight) > 0.0 or float(policy_regret_weight) > 0.0 or float(policy_admission_distill_weight) > 0.0:
            teacher_util = torch.cat([
                torch.zeros((1,), dtype=t_delta.dtype, device=t_delta.device),
                t_delta,
            ], dim=0)
            if bool(safe_positive_mask.any()):
                teacher_prob = torch.zeros_like(teacher_util)
                safe_indices = torch.where(safe_positive_mask)[0]
                safe_logits = t_delta[safe_indices] / max(float(policy_teacher_temperature), 1.0e-3)
                teacher_prob[1 + safe_indices] = torch.softmax(safe_logits, dim=0)
                teacher_prob = teacher_prob.detach()
            else:
                teacher_prob = torch.zeros_like(teacher_util)
                teacher_prob[0] = 1.0
            ranking_logits = rank_class_logits if bool(policy_decouple_admission) else admission_class_logits
            ranking_prob = torch.softmax(ranking_logits, dim=0)
            if float(policy_distill_weight) > 0.0:
                terms.append(
                    float(policy_distill_weight)
                    * F.kl_div(torch.log(ranking_prob.clamp_min(1.0e-8)), teacher_prob, reduction="sum")
                )
            if float(policy_admission_distill_weight) > 0.0:
                admission_prob = torch.softmax(admission_class_logits, dim=0)
                terms.append(
                    float(policy_admission_distill_weight)
                    * F.kl_div(torch.log(admission_prob.clamp_min(1.0e-8)), teacher_prob, reduction="sum")
                )
            if float(policy_regret_weight) > 0.0:
                expected_teacher_adv = (ranking_prob[1:] * t_delta).sum()
                oracle_teacher_adv = torch.maximum(
                    torch.zeros((), dtype=t_delta.dtype, device=t_delta.device), t_delta.max()
                )
                regret = F.relu(oracle_teacher_adv - expected_teacher_adv - float(policy_regret_margin))
                terms.append(float(policy_regret_weight) * regret.square())
        if float(pairwise_weight) > 0.0:
            pair_terms = []
            if bool(pos_mask.any()):
                pair_terms.append(F.softplus((float(rank_margin) - r_delta[pos_mask]) / tau).mean() * tau)
            if bool(neg_mask.any()):
                pair_terms.append(float(false_positive_weight) * F.softplus((r_delta[neg_mask] + float(rank_margin)) / tau).mean() * tau)
            if bool(tie_mask.any()):
                pair_terms.append(
                    float(ambiguous_group_weight)
                    * F.smooth_l1_loss(r_delta[tie_mask], t_delta[tie_mask])
                )
            if pair_terms:
                terms.append(float(pairwise_weight) * torch.stack(pair_terms).sum())
        if float(top_rank_weight) > 0.0 and bool(pos_mask.any()) and bool(neg_mask.any()):
            best_pos = r_delta[best_j]
            strongest_neg = torch.max(r_delta[neg_mask])
            terms.append(float(top_rank_weight) * F.softplus((float(rank_margin) - (best_pos - strongest_neg)) / tau) * tau)
        if float(t_adv.item()) >= float(positive_gain):
            cls_w = float(positive_group_weight)
            terms.append(float(advantage_weight) * (F.smooth_l1_loss(p_adv, t_adv) + F.softplus((float(rank_margin) - p_adv) / tau) * tau))
        elif float(t_adv.item()) <= -float(negative_gain):
            cls_w = float(negative_group_weight)
            max_pred = torch.max(p_delta)
            terms.append(float(advantage_weight) * float(false_positive_weight) * F.softplus((max_pred + float(rank_margin)) / tau) * tau)
        else:
            cls_w = float(ambiguous_group_weight)
            terms.append(0.25 * float(advantage_weight) * F.smooth_l1_loss(p_adv, t_adv))
        b = int(bid[nom].item())
        bucket_w = float(near_weight) if b == 1 else (float(contact_weight) if b == 2 else 1.0)
        group_losses.append(torch.stack(terms).sum())
        group_weights.append(max(1.0e-6, cls_w * bucket_w))
        # Pseudo-environments use only training-observable grouping plus teacher
        # state for supervision: regime, nominal severity, opportunity state, and
        # teacher-best macro.  This prevents a single train-specific severity/macro
        # pocket from dominating without peeking at calibration or test data.
        severity_value = float(target[nom].detach().item())
        severity_bin = sum(severity_value > float(x) for x in tuple(group_dro_severity_thresholds))
        state_bin = 2 if bool(pos_mask.any()) else (0 if bool(harmful_mask.all()) else 1)
        best_macro = int(mac[recs[best_j]].detach().item()) if recs.numel() else -1
        group_domains.append((b, int(severity_bin), int(state_bin), best_macro))
    if not group_losses:
        return float(point_weight) * point_loss + grad_anchor
    gl = torch.stack(group_losses)
    gw = torch.as_tensor(group_weights, dtype=gl.dtype, device=gl.device)
    erm_grouped = (gl * gw).sum() / gw.sum().clamp_min(1.0e-6)
    grouped = erm_grouped
    if bool(ordinal_evidence_batch_balanced) and evidence_proposal_records:
        if bool(ordinal_evidence_factorized_harm):
            # Balance each binary hypothesis separately.  Joint three-class
            # balancing cannot represent candidates that are both beneficial and
            # component-harmful.
            target_prob = min(0.99, max(0.51, float(ordinal_evidence_target_probability)))
            regime_terms: list[torch.Tensor] = []
            benefit_margin_terms: list[torch.Tensor] = []
            harm_margin_terms: list[torch.Tensor] = []
            balance_ids = (
                [None]
                if bool(ordinal_evidence_global_balance)
                else sorted({record[0] for record in evidence_proposal_records})
            )
            for regime_id in balance_ids:
                records = (
                    evidence_proposal_records
                    if regime_id is None
                    else [record for record in evidence_proposal_records if record[0] == regime_id]
                )
                tail_terms: list[torch.Tensor] = []
                for loss_index, label_index in ((1, 5), (2, 6)):
                    class_means: list[torch.Tensor] = []
                    for label in (False, True):
                        selected = [record[loss_index] for record in records if bool(record[label_index]) is label]
                        if selected:
                            class_means.append(torch.stack(selected).mean())
                    if class_means:
                        tail_terms.append(torch.stack(class_means).mean())
                if tail_terms:
                    regime_terms.append(torch.stack(tail_terms).mean())
                benefit_pos = [record[3] for record in records if bool(record[5])]
                harm_pos = [record[4] for record in records if bool(record[6])]
                if benefit_pos and float(ordinal_evidence_benefit_margin_weight) > 0.0:
                    benefit_margin_terms.append(F.relu(target_prob - torch.stack(benefit_pos)).mean())
                if harm_pos and float(ordinal_evidence_harm_margin_weight) > 0.0:
                    harm_margin_terms.append(F.relu(target_prob - torch.stack(harm_pos)).mean())
            balanced_terms: list[torch.Tensor] = []
            if regime_terms and float(ordinal_evidence_class_balanced_weight) > 0.0:
                balanced_terms.append(float(ordinal_evidence_class_balanced_weight) * torch.stack(regime_terms).mean())
            if benefit_margin_terms:
                balanced_terms.append(float(ordinal_evidence_benefit_margin_weight) * torch.stack(benefit_margin_terms).mean())
            if harm_margin_terms:
                balanced_terms.append(float(ordinal_evidence_harm_margin_weight) * torch.stack(harm_margin_terms).mean())
            if balanced_terms:
                balanced_objective = torch.stack(balanced_terms).sum()
                grouped = balanced_objective if bool(ordinal_evidence_balanced_replaces_erm) else grouped + balanced_objective
        else:
            target_prob = min(0.99, max(0.51, float(ordinal_evidence_target_probability)))
            regime_terms: list[torch.Tensor] = []
            benefit_margin_terms: list[torch.Tensor] = []
            harm_margin_terms: list[torch.Tensor] = []
            balance_ids = (
                [None]
                if bool(ordinal_evidence_global_balance)
                else sorted({record[0] for record in evidence_proposal_records})
            )
            for regime_id in balance_ids:
                records = (
                    evidence_proposal_records
                    if regime_id is None
                    else [record for record in evidence_proposal_records if record[0] == regime_id]
                )
                class_terms: list[torch.Tensor] = []
                for class_id in (0, 1, 2):
                    cls_records = [record for record in records if record[4] == class_id]
                    if not cls_records:
                        continue
                    class_terms.append(torch.stack([record[1] for record in cls_records]).mean())
                    if class_id == 2 and float(ordinal_evidence_benefit_margin_weight) > 0.0:
                        benefit_margin_terms.append(
                            F.relu(
                                target_prob - torch.stack([record[2] for record in cls_records])
                            ).mean()
                        )
                    if class_id == 0 and float(ordinal_evidence_harm_margin_weight) > 0.0:
                        harm_margin_terms.append(
                            F.relu(
                                target_prob - torch.stack([record[3] for record in cls_records])
                            ).mean()
                        )
                if class_terms:
                    regime_terms.append(torch.stack(class_terms).mean())
            balanced_terms: list[torch.Tensor] = []
            if regime_terms and float(ordinal_evidence_class_balanced_weight) > 0.0:
                balanced_terms.append(
                    float(ordinal_evidence_class_balanced_weight) * torch.stack(regime_terms).mean()
                )
            if benefit_margin_terms:
                balanced_terms.append(
                    float(ordinal_evidence_benefit_margin_weight) * torch.stack(benefit_margin_terms).mean()
                )
            if harm_margin_terms:
                balanced_terms.append(
                    float(ordinal_evidence_harm_margin_weight) * torch.stack(harm_margin_terms).mean()
                )
            if balanced_terms:
                balanced_objective = torch.stack(balanced_terms).sum()
                if bool(ordinal_evidence_balanced_replaces_erm):
                    # Calibrator-only target adaptation: the balanced objective is the
                    # evidence ERM, not an auxiliary term added on top of dead-zone-
                    # dominated per-group NLL.  Other explicitly enabled cross-group
                    # objectives and the residual anchor are still added below.
                    grouped = balanced_objective
                else:
                    grouped = grouped + balanced_objective
    # Cross-group, regime-local AUC surrogates for the frozen policy's selected
    # candidate.  The ordered NLL calibrates probabilities; these terms enforce
    # the ranking needed to distinguish beneficial and harmful tails across
    # scene-time groups, which v48.11 left nearly random on Contact verify.
    if evidence_policy_records:
        auc_terms: list[torch.Tensor] = []
        pair_margin = float(ordinal_evidence_pairwise_margin)
        for regime_id in sorted({x[0] for x in evidence_policy_records}):
            records = [x for x in evidence_policy_records if x[0] == regime_id]
            if len(records) < 2:
                continue
            benefit_logits = torch.stack([x[1] for x in records])
            harm_logits_batch = torch.stack([x[2] for x in records])
            cls = torch.as_tensor([x[3] for x in records], device=benefit_logits.device)
            if float(ordinal_evidence_pairwise_benefit_weight) > 0.0:
                pos = benefit_logits[cls == 2]
                neg = benefit_logits[cls != 2]
                if pos.numel() and neg.numel():
                    auc_terms.append(
                        float(ordinal_evidence_pairwise_benefit_weight)
                        * F.softplus(pair_margin - (pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
                    )
            if float(ordinal_evidence_pairwise_harm_weight) > 0.0:
                pos = harm_logits_batch[cls == 0]
                neg = harm_logits_batch[cls != 0]
                if pos.numel() and neg.numel():
                    auc_terms.append(
                        float(ordinal_evidence_pairwise_harm_weight)
                        * F.softplus(pair_margin - (pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
                    )
        if auc_terms:
            grouped = grouped + torch.stack(auc_terms).sum()
    dro_mix = min(max(float(group_dro_weight), 0.0), 1.0)
    if dro_mix > 0.0 and group_domains:
        domain_means: list[torch.Tensor] = []
        for domain in sorted(set(group_domains)):
            mask = torch.as_tensor([d == domain for d in group_domains], dtype=torch.bool, device=gl.device)
            dw = gw[mask]
            domain_means.append((gl[mask] * dw).sum() / dw.sum().clamp_min(1.0e-6))
        if domain_means:
            dm = torch.stack(domain_means)
            dro_tau = max(float(group_dro_temperature), 1.0e-3)
            robust_grouped = dro_tau * (torch.logsumexp(dm / dro_tau, dim=0) - torch.log(torch.as_tensor(float(dm.numel()), dtype=dm.dtype, device=dm.device)))
            grouped = (1.0 - dro_mix) * erm_grouped + dro_mix * robust_grouped
    calibrator_anchor = pred_logit.float().sum() * 0.0
    if evidence_calibrator_residual is not None and float(evidence_calibrator_anchor_weight) > 0.0:
        calibrator_anchor = float(evidence_calibrator_anchor_weight) * evidence_calibrator_residual.float().pow(2).mean()
    return float(point_weight) * point_loss + grouped + grad_anchor + calibrator_anchor
