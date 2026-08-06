from __future__ import annotations

import importlib
import inspect
import gc
import time
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.core.constants import MacroType
from cowp.models.root_alignment import natural_root_alignment_cost
from cowp.models.losses import paper_aligned_supervision_batch
from cowp.planning.set_preservation_selector import select_set_preservation_frontier_batch
from cowp.utils.progress import tqdm_iter
from cowp.utils.dataloader_runtime import configure_dataloader_runtime
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, witness_quality, _progress_reference_m, _trajectory_progress_m
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator
from cowp.utils.checkpoint_compat import compatible_state_dict, strip_compiled_prefix


def _load_state_dict_compatible(model, state: dict) -> None:
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    state = strip_compiled_prefix(state)
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    model_state = model.state_dict()
    compatible, _, _ = compatible_state_dict(model_state, state)
    model.load_state_dict(compatible, strict=False)


def _label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if k.startswith("cowp/") or k.startswith("map/") or k.startswith("scenario/") or k.startswith("dataset/"):
            try:
                out[k] = v[i].detach().cpu().numpy()
            except Exception:
                pass
    return out


def _method_gate_defaults(method: str, gate_mode: str) -> tuple[str, str]:
    """Map method names to a learned-offline selection variant and gate mode."""
    m = str(method or "cowp").lower()
    g = str(gate_mode or "priority").lower()
    aliases = {
        "cowp_priority": "cowp",
        "priority_ncf": "cowp",
        "p_ncf": "cowp",
        "cowp_universal": "universal_ncf",
        "hard_ncf": "universal_ncf",
        "ego_utility": "idm_lattice",
        "utility_lattice": "idm_lattice",
        "safety_only": "conventional_safety",
        "planner_only": "planner_score_only",
        "no_ncf": "planner_score_only",
    }
    m = aliases.get(m, m)
    if m == "cowp" and g == "hard":
        # The paper/code now treats COWP as priority-aware NCF.  The original
        # universal veto remains available as method=universal_ncf.
        g = "priority"
    if m == "universal_ncf":
        g = "hard"
    elif m == "soft_burden_cost_only":
        g = "soft"
    elif m in {"idm_lattice", "conventional_safety", "planner_score_only", "outcome_oracle"}:
        g = "none"
    return m, g


def _stop_like_mask(batch, cand_valid, conventional):
    import torch

    macro = batch.get("cowp/candidates/macro_type")
    is_neutral = batch.get("cowp/candidates/is_neutral")
    mask = cand_valid.clone()
    if conventional is not None:
        mask = mask & conventional
    if macro is not None:
        stop_ids = torch.as_tensor(
            [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)],
            device=macro.device,
            dtype=macro.dtype,
        )
        stop = (macro[..., None] == stop_ids).any(dim=-1)
        mask = mask & stop
    elif is_neutral is not None:
        mask = mask & is_neutral.bool()
    else:
        mask = torch.zeros_like(cand_valid)
    if is_neutral is not None:
        mask = mask | (cand_valid & conventional & is_neutral.bool())
    return mask


def _witness_probability_and_certificate(pred_witness, cfg=None):
    import torch
    cfg = cfg or {}
    pcfg = cfg.get("planning", {})
    temp = max(float(pcfg.get("witness_temperature", 1.0)), 1e-3)
    bias = float(pcfg.get("witness_logit_bias", 0.0))
    logit_prob = torch.sigmoid((pred_witness["exist_logits"].float() - bias) / temp).float()
    evidence_prob = pred_witness.get("evidential_prob")
    uncertainty = pred_witness.get("epistemic_uncertainty")
    source = str(pcfg.get("witness_probability_source", "mixed")).lower()
    if source == "logit" or not torch.is_tensor(evidence_prob):
        prob = logit_prob
    elif source == "evidential":
        prob = evidence_prob.float()
    else:
        mix = float(pcfg.get("evidential_probability_mix", 0.5))
        prob = (1.0 - mix) * logit_prob + mix * evidence_prob.float()
    unc = uncertainty.float().clamp(0.0, 1.0) if torch.is_tensor(uncertainty) else torch.zeros_like(prob)
    # Invalid numerical outputs are treated conservatively, but finite outputs are
    # no longer forced through the evidential head unless explicitly requested.
    prob = torch.nan_to_num(prob, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    unc = torch.nan_to_num(unc, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ucb = float(pcfg.get("evidential_ucb_scale", 0.0 if source == "logit" else 0.15))
    return prob, (prob + ucb * unc).clamp(0.0, 1.0), unc




def _candidate_certificate_scores(pred, scores, cfg: dict | None = None, mask=None):
    """Return calibrated candidate certificate probabilities.

    A newly added certificate head can be missing or effectively flat when a run
    resumes from an older planner/witness checkpoint.  In that failure mode the
    raw head emits 0.5/0.5/0.5 and the COWP frontier becomes indistinguishable
    from conventional_safety.  Use the learned head when it has within-scene
    spread; otherwise blend in an outcome-calibrated fallback constructed from
    the planner score and the learned Waymax outcome head.  This keeps the
    frontier outcome-calibrated while the dedicated certificate head is still
    being learned, rather than silently dropping the certificate from selection.
    """
    import torch
    pcfg = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
    scores_f = torch.nan_to_num(scores.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6)
    if mask is None:
        mask_t = torch.isfinite(scores_f)
    else:
        mask_t = mask.bool()

    ncf_logit = pred.get("candidate_ncf_logit")
    fs_logit = pred.get("candidate_false_safe_logit")
    q_logit = pred.get("candidate_quality_logit")
    has_head = torch.is_tensor(ncf_logit) and torch.is_tensor(fs_logit)
    if torch.is_tensor(ncf_logit):
        ncf_prob = torch.sigmoid(torch.nan_to_num(ncf_logit.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0))
    else:
        ncf_prob = torch.sigmoid(torch.nan_to_num(-scores_f, nan=0.0, posinf=20.0, neginf=-20.0))
    if torch.is_tensor(fs_logit):
        false_safe_prob = torch.sigmoid(torch.nan_to_num(fs_logit.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0))
    else:
        false_safe_prob = torch.sigmoid(torch.nan_to_num(scores_f, nan=0.0, posinf=20.0, neginf=-20.0))
    if torch.is_tensor(q_logit):
        quality_prob = torch.sigmoid(torch.nan_to_num(q_logit.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0))
    else:
        quality_prob = ncf_prob * (1.0 - false_safe_prob)

    # Outcome-calibrated fallback risk.  Lower planner score is better, while
    # collision/offroad/log-divergence probabilities from the outcome head are
    # direct closed-loop risk predictions.  Normalize within each scene so the
    # fallback is a ranking certificate, not an absolute probability claim.
    planner_risk = _scene_normalized_risk_torch(scores_f, mask_t, cfg)
    outcome = pred.get("outcome", {})
    if isinstance(outcome, dict) and ("collision_logit" in outcome or "offroad_logit" in outcome or "logdiv" in outcome):
        col_r = torch.sigmoid(torch.nan_to_num(outcome.get("collision_logit", torch.zeros_like(scores_f)).detach().float(), nan=0.0, posinf=20.0, neginf=-20.0))
        off_r = torch.sigmoid(torch.nan_to_num(outcome.get("offroad_logit", torch.zeros_like(scores_f)).detach().float(), nan=0.0, posinf=20.0, neginf=-20.0))
        ld_r = torch.nan_to_num(outcome.get("logdiv", torch.zeros_like(scores_f)).detach().float(), nan=0.0, posinf=50.0, neginf=0.0).clamp_min(0.0) / 10.0
        outcome_risk = _scene_normalized_risk_torch(col_r + off_r + ld_r, mask_t, cfg)
        outcome_mix = float(pcfg.get("candidate_cert_fallback_outcome_mix", 0.70))
        fallback_risk = (outcome_mix * outcome_risk + (1.0 - outcome_mix) * planner_risk).clamp(0.0, 1.0)
    else:
        fallback_risk = planner_risk.clamp(0.0, 1.0)
    fb_ncf = (1.0 - fallback_risk).clamp(0.02, 0.98)
    fb_fs = fallback_risk.clamp(0.02, 0.98)
    fb_q = (1.0 - fallback_risk).clamp(0.02, 0.98)

    allow_hybrid = bool(pcfg.get("candidate_cert_allow_hybrid_fallback", False))
    if has_head and not allow_hybrid:
        return ncf_prob.clamp(0.0, 1.0), false_safe_prob.clamp(0.0, 1.0), quality_prob.clamp(0.0, 1.0)
    if has_head:
        raw_risk = _candidate_certificate_risk(ncf_prob, false_safe_prob, quality_prob, cfg)
        raw_spread = torch.zeros(raw_risk.shape[0], device=raw_risk.device, dtype=raw_risk.dtype)
        for b in range(int(raw_risk.shape[0])):
            vals = raw_risk[b, mask_t[b]] if mask_t.ndim == raw_risk.ndim else raw_risk[b]
            if vals.numel() > 1:
                raw_spread[b] = vals.float().std(unbiased=False)
        flat = raw_spread < float(pcfg.get("candidate_cert_fallback_min_std", 2.0e-3))
        base_mix = float(pcfg.get("candidate_cert_hybrid_fallback_mix", 0.25))
        flat_mix = float(pcfg.get("candidate_cert_flat_fallback_mix", 0.90))
        mix = torch.where(flat, raw_spread.new_full(raw_spread.shape, flat_mix), raw_spread.new_full(raw_spread.shape, base_mix))
        while mix.ndim < ncf_prob.ndim:
            mix = mix.unsqueeze(-1)
        ncf_prob = (1.0 - mix) * ncf_prob + mix * fb_ncf
        false_safe_prob = (1.0 - mix) * false_safe_prob + mix * fb_fs
        quality_prob = (1.0 - mix) * quality_prob + mix * fb_q
    else:
        ncf_prob, false_safe_prob, quality_prob = fb_ncf, fb_fs, fb_q
    return ncf_prob.clamp(0.0, 1.0), false_safe_prob.clamp(0.0, 1.0), quality_prob.clamp(0.0, 1.0)


def _candidate_certificate_risk(ncf_prob, false_safe_prob, quality_prob, cfg: dict | None = None):
    """Candidate-level calibrated P-NCF risk used by both offline and online selection.

    The previous risk `(1 - P_NCF) + P_false_safe` ignored the learned quality
    head and was often nearly constant, so COWP collapsed back to conventional
    safety.  This risk is still label-compatible but gives the quality head a
    direct role in the frontier ranking.  Absolute probabilities are not trusted:
    the selector below keeps a within-scene low-risk quantile frontier.
    """
    import torch
    pcfg = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
    w_ncf = float(pcfg.get("candidate_risk_ncf_weight", 1.0))
    w_fs = float(pcfg.get("candidate_risk_false_safe_weight", 2.0))
    w_q = float(pcfg.get("candidate_risk_quality_weight", 0.75))
    return (w_ncf * (1.0 - ncf_prob.clamp(0.0, 1.0))
            + w_fs * false_safe_prob.clamp(0.0, 1.0)
            + w_q * (1.0 - quality_prob.clamp(0.0, 1.0)))

def _scene_normalized_risk_torch(x, mask, cfg: dict | None = None):
    """Normalize risk per scene and zero out flat/uninformative certificates."""
    import torch
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    mask = mask.bool()
    out = torch.zeros_like(x)
    pcfg = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
    min_spread = float(pcfg.get("decision_risk_min_spread", 1e-3))
    min_std = float(pcfg.get("decision_risk_min_std", 1e-3))
    for b in range(int(x.shape[0])):
        if not bool(mask[b].any()):
            continue
        vals = x[b, mask[b]]
        lo = vals.min()
        hi = vals.quantile(0.90) if vals.numel() > 1 else vals.max()
        span = (hi - lo).clamp_min(min_spread)
        y = ((x[b] - lo) / span).clamp(0.0, 1.0)
        spread = vals.std(unbiased=False) if vals.numel() > 1 else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        out[b] = torch.where(spread >= min_std, y, torch.zeros_like(y))
    return out


def _candidate_pressure_prior_torch(batch, cfg: dict | None, scores):
    """Torch offline counterpart of the online mechanism-aware pressure prior.

    Vectorized over candidates and time.  It is a weak, relative prior used only
    to break flat candidate-certificate ties in the quantile frontier.
    """
    import torch
    cand = batch.get("cowp/candidates/trajectory")
    cand_valid = batch.get("cowp/candidates/valid")
    if cand is None or cand_valid is None:
        return torch.zeros_like(scores)
    cand = cand.float()
    cand_valid = cand_valid.bool()
    macro = batch.get("cowp/candidates/macro_type")
    macro = torch.zeros_like(cand_valid, dtype=torch.long) if macro is None else macro.long()
    hist = batch.get("state/history", batch.get("womd/state/history"))
    if hist is None or hist.ndim != 4 or hist.shape[-1] < 10:
        return torch.zeros_like(scores)
    cur = hist[:, :, -1, :].float()
    B, K = scores.shape[:2]
    out = torch.zeros_like(scores)
    crit_idx = batch.get("cowp/critical/input_index", batch.get("cowp/critical/track_index"))
    crit_valid = batch.get("cowp/critical/valid")
    if crit_idx is None or crit_valid is None:
        return out
    crit_idx = crit_idx.long()
    crit_valid = crit_valid.bool()
    is_sdc = batch.get("state/is_sdc", batch.get("womd/state/is_sdc"))
    if is_sdc is not None and is_sdc.ndim >= 2:
        sdc_idx = torch.argmax(is_sdc.float(), dim=1)
    else:
        sdc_idx = torch.zeros(B, device=scores.device, dtype=torch.long)
    dt = float((cfg or {}).get("time", {}).get("dt", 0.1))
    H = int(cand.shape[2])
    ts = (torch.arange(H, device=scores.device, dtype=torch.float32) + 1.0) * max(dt, 1e-3)
    pressure = torch.zeros_like(scores) + 0.05
    pressure = torch.where(macro == int(MacroType.ACCELERATE_CROSS), torch.full_like(pressure, 0.75), pressure)
    pressure = torch.where(macro == int(MacroType.MERGE_AHEAD), torch.full_like(pressure, 0.90), pressure)
    pressure = torch.where((macro == int(MacroType.LANE_CHANGE_LEFT)) | (macro == int(MacroType.LANE_CHANGE_RIGHT)), torch.full_like(pressure, 0.70), pressure)
    relief = (macro == int(MacroType.YIELD)) | (macro == int(MacroType.STOP_BEFORE_CONFLICT)) | (macro == int(MacroType.NEUTRAL_EGO)) | (macro == int(MacroType.CREEP)) | (macro == int(MacroType.MERGE_BEHIND))
    pressure = torch.where(relief, pressure - 0.25, pressure)
    out = torch.clamp(pressure, min=0.0, max=1.0)
    cand_xy_all = cand[:, :, :, 0:2]
    finite_all = torch.isfinite(cand_xy_all).all(dim=-1) & cand_valid[:, :, None]
    A = int(crit_idx.shape[1])
    for b in range(B):
        si = int(sdc_idx[b].item())
        if si < 0 or si >= cur.shape[1]:
            continue
        ego = cur[b, si]
        ego_xy = ego[0:2]
        ego_yaw = ego[6] if cur.shape[-1] > 6 else torch.tensor(0.0, device=scores.device)
        ego_dir = torch.stack((torch.cos(ego_yaw), torch.sin(ego_yaw)))
        ego_lat = torch.stack((-torch.sin(ego_yaw), torch.cos(ego_yaw)))
        ego_speed = torch.clamp(ego[9] if cur.shape[-1] > 9 else torch.linalg.norm(ego[7:9]), min=0.0)
        for a in range(A):
            if not bool(crit_valid[b, a].item()):
                continue
            j = int(crit_idx[b, a].item())
            if j < 0 or j >= cur.shape[1] or j == si:
                continue
            aj = cur[b, j]
            if cur.shape[-1] > 10 and float(aj[10].item()) <= 0.5:
                continue
            aj_xy = aj[0:2]
            aj_vel = aj[7:9] if cur.shape[-1] > 8 else torch.zeros(2, device=scores.device)
            pred = aj_xy[None, :] + ts[:, None] * aj_vel[None, :]
            finite = finite_all[b] & torch.isfinite(pred).all(dim=-1)[None, :]
            d = torch.linalg.norm(cand_xy_all[b] - pred[None, :, :], dim=-1)
            d = torch.where(finite, d, torch.full_like(d, float("inf")))
            min_d = torch.min(d, dim=-1).values
            close = torch.exp(-torch.clamp(min_d, min=0.0, max=1e4) / 6.5)
            close = torch.where(torch.isfinite(min_d), close, torch.zeros_like(close))
            rel = aj_xy - ego_xy
            longitudinal = torch.dot(rel, ego_dir)
            lateral = torch.abs(torch.dot(rel, ego_lat))
            rel_speed = ego_speed - torch.dot(aj_vel, ego_dir)
            priority = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
            if bool((longitudinal >= -6.0 and longitudinal <= 45.0 and lateral <= 5.5).item()):
                priority = priority + 0.35
            if bool((longitudinal > 0.0 and rel_speed > 0.25 and (longitudinal / torch.clamp(rel_speed, min=1e-3)) <= 5.0).item()):
                priority = priority + 0.25
            if bool((torch.min(min_d) <= 6.0).item()):
                priority = priority + 0.20
            out[b] = torch.maximum(out[b], pressure[b] + 0.55 * close + priority)
    out = torch.where(cand_valid, out, torch.zeros_like(out))
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)



def _topk_frontier_mask_torch(base_mask, risk, tie_breaker=None, *, keep_frac=0.40, keep_min=1, keep_max=4, eps=1.0e-3):
    """Return an exact-cardinality low-risk frontier.

    Quantile cutoffs select every tied candidate.  When certificate/outcome risk
    is flat, that expands the frontier back to the full conventional set and
    makes COWP indistinguishable from conventional_safety.  This helper always
    returns exactly top-k candidates per scene, with a deterministic tie-breaker.
    """
    import torch

    base = base_mask.bool()
    frontier = torch.zeros_like(base)
    if not bool(base.any().detach().cpu().item()):
        return frontier
    idx = torch.where(base)[0]
    n = int(idx.numel())
    k = max(int(keep_min), int(torch.ceil(torch.tensor(float(n) * float(keep_frac), device=risk.device)).item()))
    k = min(max(k, 1), n, max(int(keep_max), 1))
    r = torch.nan_to_num(risk[idx].float(), nan=1.0, posinf=1.0, neginf=0.0)
    if tie_breaker is None:
        tb = torch.arange(n, device=risk.device, dtype=r.dtype) / max(float(n), 1.0)
    else:
        tb = torch.nan_to_num(tie_breaker[idx].float(), nan=0.0, posinf=1.0, neginf=0.0)
        if tb.numel() > 1:
            lo = tb.min(); hi = tb.max(); span = (hi - lo).clamp_min(1.0e-6)
            tb = ((tb - lo) / span).clamp(0.0, 1.0)
        else:
            tb = torch.zeros_like(tb)
    order = torch.argsort(r + float(eps) * tb, stable=True)
    frontier[idx[order[:k]]] = True
    return frontier

def _candidate_progress_torch(batch, cand_valid):
    """Approximate ego progress for guarding the least-coercive frontier.

    Pure coercion minimization tends to prefer stopped/yielding plans, which can
    make the universal NCF ablation look safe but unusable.  The progress guard
    preserves the paper idea--choose among non-coercive plans--while requiring
    those plans to remain behaviorally useful unless no useful candidate exists.
    """
    import torch

    traj = batch.get("cowp/candidates/trajectory")
    if traj is None or not torch.is_tensor(traj) or traj.ndim < 4:
        return torch.zeros_like(cand_valid, dtype=torch.float32)
    xy0 = traj[:, :, 0, :2].float()
    xy1 = traj[:, :, -1, :2].float()
    delta = torch.nan_to_num(xy1 - xy0, nan=0.0, posinf=0.0, neginf=0.0)
    # Signed longitudinal progress avoids rewarding lateral drift, wrong-way and
    # off-route candidates as the Euclidean endpoint distance did in v7.
    if traj.shape[-1] >= 3:
        yaw0 = torch.nan_to_num(traj[:, :, 0, 2].float(), nan=0.0)
        heading = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
        progress = (delta * heading).sum(dim=-1).clamp_min(0.0)
    else:
        progress = torch.linalg.norm(delta, dim=-1)
    return torch.where(cand_valid.bool(), progress, torch.zeros_like(progress))



def _candidate_action_risk_torch(batch, cand_valid, cfg=None):
    """Trajectory-level kinematic risk available in both offline and online data.

    The v5 offline path passed ``None`` to the action guard, so the advertised
    ``candidate_frontier_max_action_risk`` and final action weight were dead knobs.
    This estimate uses acceleration, jerk, and yaw-rate excess from the candidate
    trajectory itself and requires no live simulator state.
    """
    import torch

    traj = batch.get("cowp/candidates/trajectory")
    if traj is None or not torch.is_tensor(traj) or traj.ndim < 4 or traj.shape[2] < 3:
        return torch.zeros_like(cand_valid, dtype=torch.float32)
    pcfg = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
    dt = max(float((cfg or {}).get("time", {}).get("dt", 0.1)) if isinstance(cfg, dict) else 0.1, 1e-3)
    xy = torch.nan_to_num(traj[..., :2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    speed = torch.linalg.norm(torch.diff(xy, dim=2), dim=-1) / dt
    accel = torch.abs(torch.diff(speed, dim=2)) / dt if speed.shape[2] > 1 else torch.zeros_like(speed)
    jerk = torch.abs(torch.diff(accel, dim=2)) / dt if accel.shape[2] > 1 else torch.zeros_like(accel)
    if traj.shape[-1] > 2:
        yaw = torch.nan_to_num(traj[..., 2].float(), nan=0.0, posinf=0.0, neginf=0.0)
        dyaw = torch.atan2(torch.sin(torch.diff(yaw, dim=2)), torch.cos(torch.diff(yaw, dim=2)))
        yaw_rate = torch.abs(dyaw) / dt
    else:
        yaw_rate = torch.zeros_like(speed)

    def excess(x, soft, hard):
        if x.numel() == 0:
            return torch.zeros_like(cand_valid, dtype=torch.float32)
        vmax = x.amax(dim=2)
        return ((vmax - float(soft)) / max(float(hard) - float(soft), 1e-3)).clamp(0.0, 1.0)

    a = excess(accel, pcfg.get("candidate_action_accel_soft_mps2", 3.0), pcfg.get("candidate_action_accel_hard_mps2", 7.0))
    j = excess(jerk, pcfg.get("candidate_action_jerk_soft_mps3", 5.0), pcfg.get("candidate_action_jerk_hard_mps3", 15.0))
    y = excess(yaw_rate, pcfg.get("candidate_action_yaw_rate_soft_rps", 0.5), pcfg.get("candidate_action_yaw_rate_hard_rps", 1.5))
    risk = torch.maximum(a, torch.maximum(j, y))
    return torch.where(cand_valid.bool(), torch.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=0.0), torch.zeros_like(risk))

def _guard_frontier_base_torch(base, score_risk, progress, action_risk=None, *, keep_min=2, pcfg=None):
    """Keep the frontier least-coercive without collapsing progress/safety.

    The guard is intentionally relaxable: it only applies a progress, score, or
    action-risk screen when enough candidates remain for the exact-cardinality
    frontier.  This gives COWP a utility-regret bounded non-coercive frontier
    instead of a pure stop-like veto.
    """
    import torch

    pcfg = pcfg or {}
    base = base.bool()
    if not bool(base.any()):
        return base
    min_keep = max(int(keep_min), 1)
    idx = torch.where(base)[0]
    need = min(min_keep, int(idx.numel()))
    guarded = base.clone()

    # Progress guard: avoid selecting only stop/yield plans when a meaningful
    # conventional candidate exists.  If the scene itself has no progress option,
    # the guard is inactive.
    if torch.is_tensor(progress) and progress.numel() == base.numel():
        vals = torch.nan_to_num(progress.float(), nan=0.0, posinf=0.0, neginf=0.0)
        p_ref = vals[idx].max() if idx.numel() else vals.new_tensor(0.0)
        min_abs = float(pcfg.get("candidate_frontier_min_progress_m", 1.0))
        ratio = float(pcfg.get("candidate_frontier_min_progress_ratio", 0.12))
        if float(p_ref.detach().cpu().item()) > min_abs:
            p_min = max(min_abs, ratio * float(p_ref.detach().cpu().item()))
            pg = base & (vals >= p_min)
            if int(pg.sum().detach().cpu().item()) >= need:
                guarded = pg

    # Utility-regret guard in scene-normalized score space.  It prevents a very
    # low-coercion but implausible candidate from replacing all planner-preferred
    # feasible actions.  The default is loose; COWP still changes choices.
    if torch.is_tensor(score_risk) and score_risk.numel() == base.numel():
        sr = torch.nan_to_num(score_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
        slack = float(pcfg.get("candidate_frontier_score_slack", 0.85))
        best = sr[idx].min() if idx.numel() else sr.new_tensor(0.0)
        sg = base & (sr <= best + slack)
        joint = guarded & sg
        if int(joint.sum().detach().cpu().item()) >= need:
            guarded = joint

    # Kinematic guard: only active when it does not empty the frontier.
    if torch.is_tensor(action_risk) and action_risk.numel() == base.numel():
        ar = torch.nan_to_num(action_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
        max_ar = float(pcfg.get("candidate_frontier_max_action_risk", 0.90))
        ag = guarded & (ar <= max_ar)
        if int(ag.sum().detach().cpu().item()) >= need:
            guarded = ag

    return guarded



def _risk_budgeted_selection_scores_torch(
    scores,
    base_mask,
    frontier_mask,
    noncoercive_risk,
    score_risk,
    progress_shortfall,
    action_risk=None,
    rule_risk=None,
    outcome_risk=None,
    *,
    pcfg=None,
):
    """Lexicographic score for utility-regret bounded P-NCF selection.

    The v4 selector correctly made COWP different from conventional_safety, but
    it added the P-NCF risk to raw planner scores.  Raw scores can dominate the
    certificate and select the highest-utility candidate inside the frontier even
    when that candidate is still predicted coercive.  This helper makes the
    selection order explicit:

    1. build a small low-coercion frontier;
    2. keep enough utility/progress through the guard;
    3. select primarily by calibrated non-coercion risk, with planner utility only
       as a bounded tie-breaker.

    The output is still a scalar because the existing evaluator uses argmin, but
    its units are normalized decision risks rather than uncalibrated raw logits.
    """
    import torch

    pcfg = pcfg or {}
    inf = torch.full_like(scores, float("inf"))
    select = frontier_mask.bool()
    if not bool(select.any().detach().cpu().item()):
        select = base_mask.bool()
    if not bool(select.any().detach().cpu().item()):
        return torch.where(base_mask.bool(), scores, inf)

    nr = torch.nan_to_num(noncoercive_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    sr = torch.nan_to_num(score_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    ps = torch.nan_to_num(progress_shortfall.float(), nan=1.0, posinf=1.0, neginf=0.0)
    ar = torch.zeros_like(nr) if action_risk is None else torch.nan_to_num(action_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    rr = torch.zeros_like(nr) if rule_risk is None else torch.nan_to_num(rule_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)
    orr = torch.zeros_like(nr) if outcome_risk is None else torch.nan_to_num(outcome_risk.float(), nan=1.0, posinf=1.0, neginf=0.0)

    # Risk budget: if the frontier contains an obviously low-coercion subset, do
    # not let a high-utility but high-coercion member win.  If the subset would be
    # empty, keep the whole frontier so COWP does not collapse to fallback.
    budget = float(pcfg.get("candidate_selection_risk_budget", 0.18))
    min_keep = max(int(pcfg.get("candidate_selection_min_keep", 1)), 1)
    vals = nr[select]
    low = vals.min() if vals.numel() else nr.new_tensor(0.0)
    budget_mask = select & (nr <= low + budget)
    if int(budget_mask.sum().detach().cpu().item()) >= min_keep:
        select = budget_mask

    # Normalized final objective.  Certificate risk is primary; score/progress are
    # only bounded utility-regret terms.  Action/outcome/rule are shields that
    # suppress kinematic or closed-loop unsafe plans without replacing the
    # non-coercion certificate itself.
    obj = (
        nr
        + float(pcfg.get("candidate_selection_score_weight", 0.18)) * sr
        + float(pcfg.get("candidate_selection_progress_weight", 0.10)) * ps
        + float(pcfg.get("candidate_selection_action_weight", 0.80)) * ar
        + float(pcfg.get("candidate_selection_rule_weight", 0.20)) * rr
        + float(pcfg.get("candidate_selection_outcome_weight", 0.45)) * orr
    )
    return torch.where(select, obj, inf)

def _select_from_learned(
    batch,
    pred,
    *,
    witness_threshold: float = 0.5,
    bcot_risk_budget: float | None = None,
    alpha_opr: float = 0.35,
    gate_mode: str = "priority",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    cfg: dict | None = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[bool]]:
    import torch

    method, gate_mode = _method_gate_defaults(method, gate_mode)
    scores = torch.nan_to_num(pred["planner_score"].detach().float(), nan=1e6, posinf=1e6, neginf=-1e6)
    witness_prob, witness_cert, witness_uncertainty = _witness_probability_and_certificate(pred["witness"], cfg)
    witness_prob = witness_prob.detach()
    witness_cert = witness_cert.detach()
    opr = torch.nan_to_num(pred["witness"]["opr"].detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    cand_valid = batch["cowp/candidates/valid"].bool()
    conventional = batch.get("cowp/candidates/conventional_safe", cand_valid).bool()
    utility = batch.get("cowp/candidates/ego_utility_prior")
    if utility is not None:
        utility_scores = torch.nan_to_num(utility.detach().float(), nan=1e6, posinf=1e6, neginf=-1e6)
    else:
        utility_scores = scores
    cand_ncf_prob, cand_false_safe_prob, cand_quality_prob = _candidate_certificate_scores(pred, scores, cfg, cand_valid)
    candidate_cert_risk = _candidate_certificate_risk(cand_ncf_prob, cand_false_safe_prob, cand_quality_prob, cfg)
    transport_risk = pred.get("candidate_transport_risk")
    if torch.is_tensor(transport_risk):
        transport_risk = torch.nan_to_num(
            transport_risk.detach().float(), nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    else:
        transport_risk = None
    transport_uncertainty = pred.get("candidate_transport_uncertainty")
    if torch.is_tensor(transport_uncertainty):
        transport_uncertainty = torch.nan_to_num(
            transport_uncertainty.detach().float(), nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    else:
        transport_uncertainty = torch.zeros_like(scores)
    transport_severe = pred.get("candidate_transport_severe_prob")
    if torch.is_tensor(transport_severe):
        transport_severe = torch.nan_to_num(
            transport_severe.detach().float(), nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
    else:
        transport_severe = torch.zeros_like(scores)
    pressure_prior = _candidate_pressure_prior_torch(batch, cfg, scores)
    rule_risk = batch.get("cowp/candidates/rule_risk")
    if rule_risk is not None:
        rule_risk = torch.nan_to_num(rule_risk.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    else:
        rule_risk = torch.zeros_like(scores)
    cert_decision_risk = _scene_normalized_risk_torch(candidate_cert_risk, cand_valid, cfg)
    pressure_decision_risk = _scene_normalized_risk_torch(pressure_prior, cand_valid, cfg)
    rule_decision_risk = _scene_normalized_risk_torch(rule_risk, cand_valid, cfg)
    action_risk = _candidate_action_risk_torch(batch, cand_valid, cfg)
    action_decision_risk = _scene_normalized_risk_torch(action_risk, cand_valid, cfg)
    outcome = pred.get("outcome", {})
    if isinstance(outcome, dict) and float(outcome_risk_penalty) > 0.0:
        col_r = torch.sigmoid(outcome.get("collision_logit", torch.zeros_like(scores)).detach().float()).clamp(0.0, 1.0)
        off_r = torch.sigmoid(outcome.get("offroad_logit", torch.zeros_like(scores)).detach().float()).clamp(0.0, 1.0)
        # Probability union stays in [0,1].  Do not use log-divergence here: the
        # attached v5 caches contain no finite/non-zero log-divergence labels.
        outcome_risk = torch.nan_to_num(1.0 - (1.0 - col_r) * (1.0 - off_r), nan=1.0, posinf=1.0, neginf=0.0)
    else:
        outcome_risk = torch.zeros_like(scores)
    outcome_decision_risk = _scene_normalized_risk_torch(outcome_risk, cand_valid, cfg)
    crit_mask = batch.get("cowp/critical/valid")
    if crit_mask is not None and witness_prob.ndim == 3:
        cm = crit_mask.bool()[:, None, :]
        witness_prob = torch.where(cm, witness_prob, torch.zeros_like(witness_prob))
        witness_cert = torch.where(cm, witness_cert, torch.zeros_like(witness_cert))
        witness_uncertainty = torch.where(cm, witness_uncertainty, torch.ones_like(witness_uncertainty))
        opr = torch.where(cm, opr, torch.ones_like(opr))

    # Offline upper-bound baseline when attached Waymax candidate outcomes exist.
    if method == "outcome_oracle":
        rv = batch.get("waymax/candidate_rollout_valid", cand_valid).bool()
        col = batch.get("waymax/candidate_collision")
        off = batch.get("waymax/candidate_offroad")
        ld = batch.get("waymax/candidate_log_divergence")
        cost = torch.zeros_like(scores)
        if col is not None:
            cost = cost + col.float() * 10.0
        if off is not None:
            cost = cost + off.float() * 10.0
        if ld is not None:
            cost = cost + torch.nan_to_num(ld.float(), nan=50.0, posinf=50.0, neginf=0.0) / 10.0
        accepted = cand_valid & conventional & rv
        adjusted_scores = cost + 0.01 * utility_scores
    # Internal baselines that do not use the coercion witness.
    elif method == "idm_lattice":
        accepted = cand_valid & conventional
        adjusted_scores = utility_scores
    elif method == "conventional_safety":
        accepted = cand_valid & conventional
        adjusted_scores = scores
    elif method == "planner_score_only":
        accepted = cand_valid
        adjusted_scores = scores
    else:
        adjusted_scores = scores
        if gate_mode in {"none", "off"}:
            accepted = cand_valid & conventional
        elif gate_mode == "hard":
            predicted_bad = (witness_cert >= witness_threshold).any(dim=-1)
            accepted = cand_valid & conventional & ~predicted_bad
            accepted = accepted & (opr.min(dim=-1).values >= float(alpha_opr))
        else:
            # Priority-aware P-NCF gate.  Low OPR is evidence of option collapse,
            # but it is not a protected-priority claim by itself.  The previous
            # implementation used ``priority_proxy > 0`` after mixing in an OPR
            # indicator, which collapsed priority-aware COWP into the universal NCF
            # veto and produced all-fallback learned-offline/Waymax behavior.
            if "priority_claim_logits" in pred and torch.is_tensor(pred["priority_claim_logits"]):
                learned_priority = torch.sigmoid(pred["priority_claim_logits"].detach().float())
            else:
                learned_priority = torch.zeros_like(witness_prob)
            # Offline eval has no live Waymax state, so use cached witness rho as a
            # physically grounded priority anchor.  Online evaluation uses the live
            # priority heuristic in policy_wrapper.py.  This keeps offline/online
            # COWP semantics aligned instead of relying on an uncalibrated learned
            # priority head alone.
            rho = batch.get("cowp/witness/rho")
            if rho is not None and torch.is_tensor(rho):
                rho_l = rho.long()
                # PriorityRelation: ego=1, agent=2, equal/negotiated=3.  OPR is
                # an outcome of option transport, not evidence that an agent owns
                # right-of-way; mixing it into priority was circular and made the
                # purported priority-aware gate approach universal NCF.
                rule_priority = torch.where(
                    rho_l == 2, torch.ones_like(witness_prob),
                    torch.where(
                        rho_l == 3, torch.full_like(witness_prob, 0.65),
                        torch.where(rho_l == 1, torch.full_like(witness_prob, 0.10),
                                    torch.full_like(witness_prob, 0.25)),
                    ),
                )
            else:
                rule_priority = torch.full_like(witness_prob, 0.25)
            # Protected semantics are symbolic, not a soft score.  Agent-priority
            # (rho=2) and equal/negotiated (rho=3) relations belong to the protected
            # set by definition and must not be attenuated by an uncalibrated learned
            # head.  The learned head is used only when rho is unknown.
            if rho is not None and torch.is_tensor(rho):
                rule_protected = (rho_l == 2) | (rho_l == 3)
                rule_ego_priority = rho_l == 1
                priority_proxy = torch.where(
                    rule_protected,
                    torch.ones_like(learned_priority),
                    torch.where(rule_ego_priority, torch.zeros_like(learned_priority), learned_priority),
                ).clamp(0.0, 1.0)
                priority_claim = rule_protected | ((~rule_ego_priority) & (rho_l != 0) & (priority_proxy >= float(priority_hard_threshold)))
                # For PAD/unknown entries, fall back to the learned claim.
                priority_claim = torch.where(rho_l == 0, learned_priority >= float(priority_hard_threshold), priority_claim)
            else:
                priority_proxy = learned_priority.clamp(0.0, 1.0)
                priority_claim = priority_proxy >= float(priority_hard_threshold)
            pcfg_gate = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
            hard_max_unc = float(pcfg_gate.get("set_transport_hard_max_uncertainty", 0.40))
            opr_ucb_scale = float(pcfg_gate.get("set_transport_opr_ucb_scale", 0.50))
            confident_pair = witness_uncertainty <= hard_max_unc
            opr_upper = (opr + opr_ucb_scale * witness_uncertainty).clamp(0.0, 1.0)
            # v10 used an any/max reduction over up to six critical agents.  Even
            # with strong pair AUPRC, one moderate false positive rejected an
            # otherwise non-coercive candidate, limiting NCF recall to 0.14.
            # v11 applies the paper's object directly: a candidate-level budget on
            # transported option deficit, while retaining an explicit severe-pair
            # veto for high-confidence protected-priority violations.
            severe_pair_bad = ((witness_cert >= float(secondary_witness_threshold))
                               & (opr_upper <= float(secondary_opr_alpha))
                               & priority_claim & confident_pair).any(dim=-1)
            pcfg_gate = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
            transport_gate_mode = str(pcfg_gate.get("candidate_transport_gate_mode", "budget")).lower()
            if transport_risk is not None and transport_gate_mode not in {"pairmax", "pair_max", "legacy"}:
                # Pair witness probability and candidate BCOT risk have different
                # semantics/calibration.  v11 reused witness_threshold as the BCOT
                # budget, making both the sweep and online gate uninterpretable.
                configured_budget = float(pcfg_gate.get("candidate_transport_budget", 0.35))
                transport_budget = configured_budget if bcot_risk_budget is None else float(bcot_risk_budget)
                transport_budget = min(max(transport_budget, 0.02), 0.98)
                transport_ucb_scale = float(pcfg_gate.get("candidate_transport_ucb_scale", 0.25))
                transport_ucb = (transport_risk + transport_ucb_scale * transport_uncertainty).clamp(0.0, 1.0)
                primary_bad = transport_ucb >= transport_budget
                severe_prob_threshold = float(pcfg_gate.get("candidate_transport_severe_threshold", 0.80))
                aggregate_severe_hard = bool(pcfg_gate.get("candidate_transport_aggregate_severe_hard_veto", False))
                severe_bad = severe_pair_bad | (
                    aggregate_severe_hard & (transport_severe >= severe_prob_threshold)
                )
                option_bad = torch.zeros_like(primary_bad)
            else:
                primary_bad = ((witness_cert >= witness_threshold) & priority_claim & confident_pair).any(dim=-1)
                option_bad = ((opr_upper < float(alpha_opr)) & priority_claim & confident_pair).any(dim=-1)
                severe_bad = severe_pair_bad
            uncertain_mix = float(pcfg_gate.get("set_transport_uncertain_penalty", 0.25))
            pair_soft = witness_prob + uncertain_mix * witness_uncertainty
            pair_penalty = (pair_soft * priority_proxy).amax(dim=-1) + (torch.relu(float(alpha_opr) - opr) * priority_proxy).amax(dim=-1)
            penalty = transport_risk if transport_risk is not None else pair_penalty
            cert_penalty = float((cfg or {}).get("planning", {}).get("candidate_certificate_penalty", 1.0))
            pressure_penalty = float((cfg or {}).get("planning", {}).get("candidate_pressure_prior_penalty", 0.75))
            rule_penalty = float((cfg or {}).get("planning", {}).get("candidate_rule_risk_penalty", 1.25))
            transport_pure = bool(pcfg_gate.get("candidate_transport_pure_selector", True)) and transport_risk is not None
            if transport_pure:
                transport_decision = (
                    transport_risk
                    + float(pcfg_gate.get("candidate_transport_uncertainty_penalty", 0.15)) * transport_uncertainty
                ).clamp(0.0, 1.0)
                adjusted_scores = (
                    scores
                    + float(soft_ncf_penalty) * transport_decision
                    + rule_penalty * rule_decision_risk
                    + float(pcfg_gate.get("candidate_action_risk_penalty", 1.0)) * action_decision_risk
                    + float(outcome_risk_penalty) * outcome_decision_risk
                )
            else:
                adjusted_scores = (
                    scores
                    + float(soft_ncf_penalty) * penalty
                    + cert_penalty * cert_decision_risk
                    + pressure_penalty * pressure_decision_risk
                    + rule_penalty * rule_decision_risk
                    + float(outcome_risk_penalty) * outcome_decision_risk
                )
            if gate_mode == "soft":
                accepted = cand_valid & conventional
            else:
                accepted = cand_valid & conventional & ~primary_bad & ~option_bad & ~severe_bad & (outcome_risk <= float(outcome_risk_threshold))

    # ``accepted`` is the semantic certificate set.  The Pareto frontier below is
    # only a shortlist used to choose one plan; conflating the two makes certificate
    # recall and accepted-rate depend on an arbitrary top-k implementation detail.
    certificate_accepted = accepted.clone()
    selection_mask = accepted

    if method == "cowp" and gate_mode in {"priority", "soft"}:
        # Candidate-calibrated P-NCF frontier.  The pair witness is still used as
        # explanatory evidence, but selection is anchored by a candidate-level NCF
        # certificate trained from the dataset's noncoercive_feasible/false_safe
        # labels.  This avoids the max-pair saturation observed in the current run.
        pcfg = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
        physical_ok = (
            (action_risk <= float(pcfg.get("candidate_hard_max_action_risk", 0.45)))
            & (rule_risk <= float(pcfg.get("candidate_hard_max_rule_risk", 0.70)))
            & (outcome_risk <= float(outcome_risk_threshold))
        )
        # v7 overwrote the semantic hard gate here.  Preserve accepted so witness
        # threshold / OPR / severe-witness vetoes actually constrain the frontier.
        frontier_base = accepted & physical_ok
        pair_risk = penalty if "penalty" in locals() else witness_prob.amax(dim=-1) + torch.relu(float(alpha_opr) - opr).amax(dim=-1)
        pair_mix = float((cfg or {}).get("planning", {}).get("candidate_pair_risk_mix", 0.20))
        pressure_mix = float((cfg or {}).get("planning", {}).get("candidate_pressure_prior_mix", 0.35))
        rule_mix = float((cfg or {}).get("planning", {}).get("candidate_rule_risk_mix", 1.0))
        outcome_mix = float((cfg or {}).get("planning", {}).get("candidate_outcome_risk_mix", outcome_risk_penalty))
        shield_mix = float((cfg or {}).get("planning", {}).get("candidate_frontier_shield_tie_mix", 0.08))
        # Two-level frontier: the frontier itself is a non-coercion certificate
        # layer.  Closed-loop rule/outcome risks act as a weak shield/tie-breaker
        # instead of replacing the paper's P-NCF risk.
        transport_pure = bool(pcfg.get("candidate_transport_pure_selector", True)) and transport_risk is not None
        if transport_pure:
            noncoercive_risk = (
                transport_risk
                + float(pcfg.get("candidate_transport_uncertainty_penalty", 0.15)) * transport_uncertainty
            ).clamp(0.0, 1.0)
            frontier_ncf_prob = (1.0 - transport_risk).clamp(0.0, 1.0)
            frontier_false_safe_prob = transport_risk
        else:
            noncoercive_risk = cert_decision_risk + pair_mix * pair_risk + pressure_mix * pressure_decision_risk
            frontier_ncf_prob = cand_ncf_prob
            frontier_false_safe_prob = cand_false_safe_prob
        action_mix = float((cfg or {}).get("planning", {}).get("candidate_action_risk_mix", 1.0))
        shield_risk = rule_mix * rule_decision_risk + action_mix * action_decision_risk + outcome_mix * outcome_decision_risk
        risk = noncoercive_risk + shield_mix * shield_risk
        progress = _candidate_progress_torch(batch, cand_valid)
        score_decision_risk = _scene_normalized_risk_torch(scores, cand_valid, cfg)
        prog_ref = progress.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
        progress_shortfall = (1.0 - (progress / prog_ref).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        min_ncf = float(pcfg.get("candidate_min_ncf_prob", 0.05))
        max_fs = float(pcfg.get("candidate_max_false_safe_prob", 0.95))
        keep_frac = float(pcfg.get("candidate_frontier_keep_fraction", 0.40))
        keep_min = int(pcfg.get("candidate_frontier_min_keep", 1))
        keep_max = int(pcfg.get("candidate_frontier_max_keep", 4))
        frontier, frontier_scores, _pareto_counts = select_set_preservation_frontier_batch(
            scores=scores,
            base_mask=frontier_base,
            noncoercive_risk=noncoercive_risk,
            score_risk=score_decision_risk,
            progress=progress,
            progress_shortfall=progress_shortfall,
            action_risk=action_decision_risk,
            rule_risk=rule_decision_risk,
            outcome_risk=outcome_decision_risk,
            ncf_probability=frontier_ncf_prob,
            false_safe_probability=frontier_false_safe_prob,
            cfg=pcfg,
        )
        has_frontier = frontier.any(dim=1)
        selection_mask = torch.where(has_frontier[:, None], frontier, certificate_accepted)
        adjusted_scores = torch.where(has_frontier[:, None], frontier_scores, adjusted_scores)

    selected: list[int] = []
    certificate_masks: list[np.ndarray] = []
    shortlist_masks: list[np.ndarray] = []
    fallback_flags: list[bool] = []
    B = scores.shape[0]
    # A fallback is explicitly uncertified.  Rank it by a robust least-coercive
    # objective rather than restricting to STOP/YIELD macros, which can transfer a
    # large burden to a close rear or merging vehicle.  Stop-like behavior is only a
    # weak tie-breaker after transport/rule/action risk.
    pcfg_fb = (cfg or {}).get("planning", {}) if isinstance(cfg, dict) else {}
    transport_ucb_fb = (
        (transport_risk if transport_risk is not None else candidate_cert_risk.clamp(0.0, 1.0))
        + float(pcfg_fb.get("fallback_transport_ucb_scale", pcfg_fb.get("candidate_transport_ucb_scale", 0.25))) * transport_uncertainty
    ).clamp(0.0, 1.0)
    stop_like_all = _stop_like_mask(batch, cand_valid, conventional)
    fallback_score = (
        float(pcfg_fb.get("fallback_transport_weight", 2.5)) * transport_ucb_fb
        + float(pcfg_fb.get("fallback_rule_weight", 1.0)) * rule_decision_risk
        + float(pcfg_fb.get("fallback_action_weight", 0.75)) * action_decision_risk
        + float(pcfg_fb.get("fallback_pressure_weight", 0.75)) * pressure_decision_risk
        + float(pcfg_fb.get("fallback_outcome_weight", 0.50)) * outcome_decision_risk
        + float(pcfg_fb.get("fallback_utility_weight", 0.05)) * _scene_normalized_risk_torch(utility_scores, cand_valid, cfg)
        - float(pcfg_fb.get("fallback_stop_like_bonus", 0.05)) * stop_like_all.float()
    )

    for b in range(B):
        mask = selection_mask[b]
        used_fallback = False
        if mask.any():
            masked = torch.where(mask, adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
            selected.append(int(torch.argmin(masked).item()))
        else:
            used_fallback = True
            if str(offline_fallback).lower() in {"stop_like", "least_coercive", "certificate_aware"}:
                pool = cand_valid[b] & conventional[b]
                if not pool.any():
                    pool = cand_valid[b]
                if pool.any():
                    masked = torch.where(pool, fallback_score[b], torch.full_like(fallback_score[b], float("inf")))
                    selected.append(int(torch.argmin(masked).item()))
                else:
                    selected.append(-1)
            else:
                selected.append(-1)
        certificate_masks.append(certificate_accepted[b].detach().cpu().numpy())
        shortlist_masks.append(mask.detach().cpu().numpy())
        fallback_flags.append(bool(used_fallback))
    return selected, certificate_masks, shortlist_masks, fallback_flags


def _average_precision_binary(score: np.ndarray, target: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    if score.size == 0 or int(target.sum()) == 0:
        return 0.0
    order = np.argsort(-score)
    y = target[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.maximum(np.arange(1, len(y) + 1, dtype=np.float64), 1.0)
    return float((precision * y).sum() / max(float(y.sum()), 1.0))


def _root_low_safe_target_eval(batch, mode_count: int):
    """Build the explicit per-natural-root low-burden safe-response target.

    This mirrors the training target but lives in evaluation code deliberately:
    the headline mechanism metric must be computed from cache labels rather than
    from candidate-level false-safe labels or a learned classifier.
    """
    import torch

    if int(mode_count) <= 0:
        return None
    soft = batch.get("cowp/transport/root_low_safe_score")
    if soft is not None:
        return soft.float()[..., : int(mode_count)].clamp(0.0, 1.0).ge(0.35)
    required = (
        "cowp/response/valid",
        "cowp/response/is_safe",
        "cowp/response/is_low_burden",
        "cowp/transport/response_root_index",
    )
    if any(k not in batch for k in required):
        return None
    low_safe = (
        batch["cowp/response/valid"].bool()
        & batch["cowp/response/is_safe"].bool()
        & batch["cowp/response/is_low_burden"].bool()
    )
    root = batch["cowp/transport/response_root_index"].long()
    in_range = (root >= 0) & (root < int(mode_count))
    target = torch.zeros(
        *root.shape[:-1], int(mode_count),
        device=root.device, dtype=torch.float32,
    )
    target.scatter_add_(
        -1, root.clamp(0, int(mode_count) - 1), (low_safe & in_range).float()
    )
    return target.gt(0.0)


def _root_transport_eval_arrays(pred, batch):
    """Align unordered predicted natural roots to GT and return RIOT metrics.

    Returns direct and auxiliary root-recovery scores together with all-valid and
    conflict-conditioned masks.  Natural modes are an unordered set, therefore
    raw decoder indices are not compared directly.
    """
    import torch

    cert = pred.get("set_certificate")
    natural = pred.get("natural")
    if not isinstance(cert, dict) or not isinstance(natural, dict):
        return None
    direct = cert.get("root_transport_exist")
    auxiliary = cert.get("response_root_exist_aux")
    pred_traj = natural.get("traj")
    gt_traj = batch.get("cowp/natural/traj")
    mode_valid = batch.get("cowp/transport/mode_valid")
    mode_conflict = batch.get("cowp/transport/mode_conflict")
    if not all(torch.is_tensor(x) for x in (direct, pred_traj, gt_traj, mode_valid, mode_conflict)):
        return None
    if direct.ndim != 4 or pred_traj.ndim != 5 or gt_traj.ndim != 5:
        return None

    with torch.no_grad():
        # Source-aware multi-horizon alignment matches the training target and
        # prevents geometric cross-source root swaps.
        alignment_cost, pair_ade = natural_root_alignment_cost(
            pred_traj,
            gt_traj,
            pred_source_logits=natural.get("source_logits"),
            gt_source=batch.get("cowp/natural/source"),
        )
        assignment = alignment_cost.argmin(dim=2).long()
        natural_valid = batch.get("cowp/natural/valid")
        if torch.is_tensor(natural_valid):
            assignment = torch.where(
                natural_valid.bool(), assignment, torch.zeros_like(assignment)
            )
        idx = assignment[:, None, :, :].expand(
            direct.shape[0], direct.shape[1], assignment.shape[1], assignment.shape[2]
        )
        direct_aligned = torch.gather(direct.float(), -1, idx).clamp(0.0, 1.0)
        aux_aligned = None
        if torch.is_tensor(auxiliary) and auxiliary.shape == direct.shape:
            aux_aligned = torch.gather(auxiliary.float(), -1, idx).clamp(0.0, 1.0)
        target = _root_low_safe_target_eval(batch, assignment.shape[-1])
        if target is None:
            return None
        candidate_valid = batch["cowp/candidates/valid"].bool()[:, :, None, None]
        critical_valid = batch["cowp/critical/valid"].bool()[:, None, :, None]
        all_mask = mode_valid.bool() & candidate_valid & critical_valid
        target_confidence = batch.get("cowp/transport/root_target_confidence")
        if torch.is_tensor(target_confidence):
            all_mask = all_mask & (target_confidence.float()[..., : all_mask.shape[-1]] >= 0.25)
        conflict_mask = all_mask & mode_conflict.bool()
        rho = batch.get("cowp/witness/rho")
        if torch.is_tensor(rho):
            protected = ((rho.long() == 2) | (rho.long() == 3))[:, :, :, None]
            priority_conflict_mask = conflict_mask & protected
        else:
            priority_conflict_mask = torch.zeros_like(conflict_mask)
        nearest = pair_ade.min(dim=2).values
        nearest_mask = natural_valid.bool() if torch.is_tensor(natural_valid) else all_mask.any(dim=1)
        assignment_ade_sum = float(nearest[nearest_mask].sum().item()) if nearest_mask.any() else 0.0
        assignment_ade_count = int(nearest_mask.sum().item())
    return {
        "direct": direct_aligned.detach().cpu().numpy(),
        "aux": aux_aligned.detach().cpu().numpy() if aux_aligned is not None else None,
        "target": target.detach().cpu().numpy(),
        "all_mask": all_mask.detach().cpu().numpy(),
        "conflict_mask": conflict_mask.detach().cpu().numpy(),
        "priority_conflict_mask": priority_conflict_mask.detach().cpu().numpy(),
        "assignment_ade_sum": assignment_ade_sum,
        "assignment_ade_count": assignment_ade_count,
    }


def _binary_recall_at(score: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    positives = int(target.sum())
    if positives <= 0:
        return 0.0
    return float(((score >= float(threshold)) & target).sum() / positives)


def _ranking_pair_accuracy(scores: np.ndarray, ncf: np.ndarray, false_safe: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    good = 0
    total = 0
    for b in range(scores.shape[0]):
        pos = np.where(valid[b] & ncf[b])[0]
        neg = np.where(valid[b] & false_safe[b])[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        total += int(len(pos) * len(neg))
        good += int((scores[b, pos[:, None]] < scores[b, neg[None, :]]).sum())
    return good, total


def offline_candidate_eval(labels_dir: str | Path, cfg: dict, method: str = "cowp", progress: bool = True) -> dict[str, float]:
    labels = []
    selected = []
    planner = planner_for_method(method, cfg)
    paths = sorted(Path(labels_dir).glob("*.npz"))
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc=f"Offline COWP label eval ({method})", unit="file")
    for path in iterator:
        with np.load(path, allow_pickle=True) as data:
            label = {k: data[k] for k in data.files}
        decision = planner.select_from_labels(label)
        labels.append(label)
        selected.append(decision.candidate_index)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(selected=decision.candidate_index, reason=decision.reason[:24], refresh=True)
    return metrics_from_labels(selected, labels)



_EVAL_LABEL_KEYS = {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/macro_type",
    "cowp/candidates/ego_utility_prior",
    "cowp/candidates/is_neutral",
    "cowp/candidates/is_logged",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/critical/valid",
    "cowp/natural/beta",
    "cowp/witness/exists",
    "cowp/witness/token",
    "cowp/witness/burden_total",
    "cowp/witness/min_safe_burden",
    "cowp/witness/tail_burden_excess",
    "cowp/witness/root_min_safe_burden",
    "cowp/witness/opr",
    "cowp/witness/conflict_interval",
    "cowp/witness/rho",
    "waymax/candidate_rollout_valid",
    "waymax/candidate_collision",
    "waymax/candidate_offroad",
    "waymax/candidate_log_divergence",
}


def _slim_label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    """Copy only tensors needed by learned-offline metrics.

    The previous learned-offline evaluator used the generic label extractor while
    loading a non-stage-filtered dataset.  On real caches this can retain large
    response/natural/waymax tensors for every scene until the end of evaluation,
    which explains a host-RAM OOM kill even when GPU memory looks fine.
    """
    out: dict[str, np.ndarray] = {}
    for k in _EVAL_LABEL_KEYS:
        v = batch.get(k)
        if v is None:
            continue
        try:
            out[k] = v[i].detach().cpu().numpy()
        except Exception:
            pass
    return out


class _LabelMetricAccumulator:
    def __init__(self, *, beta_default: float = 0.65) -> None:
        self.beta_default = float(beta_default)
        self.n = 0
        self.cf_count = 0
        self.false_safe = 0
        self.cbs_sum = 0.0
        self.cbs_count = 0
        self.opr_sum = 0.0
        self.opr_count = 0
        self.hbcr = 0
        self.collision_or_offroad = 0
        self.fallback_count = 0
        self.progress_m_sum = 0.0
        self.progress_norm_sum = 0.0

    def add(self, k: int, label: dict[str, np.ndarray], *, fallback_used: bool = False) -> None:
        self.n += 1
        ref_progress = max(_progress_reference_m(label), 1e-6)
        if fallback_used:
            self.fallback_count += 1
        if k < 0:
            if not fallback_used:
                self.fallback_count += 1
            return
        valid = np.asarray(label.get("cowp/candidates/valid", []), dtype=bool)
        if k >= len(valid) or not bool(valid[k]):
            if not fallback_used:
                self.fallback_count += 1
            return
        conv_arr = np.asarray(label.get("cowp/candidates/conventional_safe", valid), dtype=bool)
        conv = bool(conv_arr[k]) if k < len(conv_arr) else bool(valid[k])
        self.collision_or_offroad += int(not conv)
        traj = np.asarray(label["cowp/candidates/trajectory"])[k]
        p_m = _trajectory_progress_m(traj)
        self.progress_m_sum += float(p_m)
        self.progress_norm_sum += float(np.clip(p_m / ref_progress, 0.0, 1.0))

        crit = np.asarray(label.get("cowp/critical/valid", []), dtype=bool)
        if crit.size == 0:
            return
        wit = np.asarray(label.get("cowp/witness/exists", np.zeros((len(valid), len(crit)), dtype=bool)), dtype=bool)[k] & crit
        self.cf_count += int(conv)
        self.false_safe += int(conv and np.any(wit))
        if np.any(crit):
            burden = np.asarray(label.get("cowp/witness/burden_total", np.zeros((len(valid), len(crit)), dtype=np.float32)), dtype=np.float32)[k]
            opr = np.asarray(label.get("cowp/witness/opr", np.ones((len(valid), len(crit)), dtype=np.float32)), dtype=np.float32)[k]
            self.cbs_sum += float(np.nanmax(np.nan_to_num(burden[crit], nan=0.0, posinf=0.0, neginf=0.0)))
            self.cbs_count += 1
            self.opr_sum += float(np.nanmean(np.nan_to_num(opr[crit], nan=0.0, posinf=1.0, neginf=0.0)))
            self.opr_count += 1
            min_safe = np.asarray(label.get("cowp/witness/min_safe_burden", burden[None, ...].repeat(len(valid), axis=0)), dtype=np.float32)[k]
            beta = np.asarray(label.get("cowp/natural/beta", np.full(len(crit), self.beta_default, dtype=np.float32)), dtype=np.float32)
            if beta.shape[0] < crit.shape[0]:
                beta = np.pad(beta, (0, crit.shape[0] - beta.shape[0]), constant_values=self.beta_default)
            self.hbcr += int(np.any(min_safe[crit] > beta[: crit.shape[0]][crit]))

    def finish(self) -> dict[str, float]:
        return {
            "CR": float(self.collision_or_offroad / max(self.n, 1)),
            "EP": float(self.progress_norm_sum / max(self.n, 1)),
            "EP_m": float(self.progress_m_sum / max(self.n, 1)),
            "FallbackRate": float(self.fallback_count / max(self.n, 1)),
            "FSR": float(self.false_safe / max(self.cf_count, 1)),
            "CBS": float(self.cbs_sum / max(self.cbs_count, 1)),
            "OPR": float(self.opr_sum / max(self.opr_count, 1)),
            "HBCR": float(self.hbcr / max(self.n, 1)),
        }


def _align_candidate_vector(values: np.ndarray, length: int, *, fill_value=False, dtype=None) -> np.ndarray:
    """Return a 1-D candidate vector with exactly ``length`` entries.

    Learned external baselines may score only the first ``max_candidates``
    candidates (for example 30), while cached COWP labels keep the full padded
    candidate table (typically 64 slots).  Metric accumulation must therefore
    treat unscored tail candidates as not accepted instead of relying on numpy
    broadcasting.  If a vector is longer than the label support, truncate it to
    the support used by the labels.
    """
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    length = int(length)
    if length <= 0:
        return np.asarray([], dtype=arr.dtype if dtype is None else dtype)
    if arr.size == length:
        return arr
    if arr.size > length:
        return arr[:length]
    pad = np.full(length - arr.size, fill_value, dtype=arr.dtype if dtype is None else dtype)
    return np.concatenate([arr, pad], axis=0)


class _LearnedMetricsAccumulator:
    def __init__(self, *, beta_default: float = 0.65) -> None:
        self.label_metrics = _LabelMetricAccumulator(beta_default=beta_default)
        self.witness_sums: dict[str, float] = {}
        self.witness_count = 0
        self.selected_total = 0
        self.selected_ncf = 0
        self.selected_false_safe = 0
        self.selected_conventional = 0
        self.selected_priority_eligible = 0
        self.selected_priority_false_safe = 0
        self.accepted_total = 0
        self.valid_total = 0
        self.accepted_ncf = 0
        self.total_ncf = 0
        self.accepted_false_safe = 0
        self.total_false_safe = 0
        self.accepted_priority_total = 0
        self.accepted_priority_ncf = 0
        self.total_priority_ncf = 0
        self.accepted_priority_false_safe = 0
        self.total_priority_false_safe = 0
        self.selected_waymax_valid = 0
        self.selected_waymax_collision = 0
        self.selected_waymax_offroad = 0
        self.selected_waymax_logdiv_sum = 0.0
        self.selected_waymax_logdiv_count = 0
        self.cert_selected_count = 0
        self.cert_selected_ncf_sum = 0.0
        self.cert_selected_fs_sum = 0.0
        self.cert_selected_quality_sum = 0.0
        self.cert_selected_risk_sum = 0.0
        self.cert_selected_pressure_sum = 0.0
        self.cert_accepted_count = 0
        self.cert_accepted_risk_sum = 0.0
        self.cert_accepted_pressure_sum = 0.0
        # Scene-level coverage decomposition. Candidate-level acceptance alone
        # cannot tell whether fallback is caused by proposal coverage or by an
        # over-restrictive certificate/selector.
        self.scene_any_valid = 0
        self.scene_any_conventional_safe = 0
        self.scene_any_ncf = 0
        self.scene_conventional_without_ncf = 0
        self.scene_any_accepted = 0
        self.scene_any_accepted_ncf = 0
        self.scene_any_priority_ncf = 0
        self.scene_any_priority_eligible = 0
        self.scene_priority_eligible_without_ncf = 0
        self.scene_any_accepted_priority_ncf = 0
        self.shortlist_total = 0
        self.scene_any_shortlist = 0
        self.scene_any_shortlist_ncf = 0
        self.fallback_selected_total = 0
        self.fallback_selected_priority_false_safe = 0
        self.fallback_selected_priority_eligible = 0
        self.fallback_selected_with_ncf_available = 0
        self.scene_ncf_available_selected_ncf = 0
        # Per-scene protected burden-tail exposure.  Keep the samples so the
        # reported upper-tail CVaR is an actual empirical tail statistic rather
        # than a mean that is merely named CVaR.
        self.protected_bte_values: list[float] = []
        self.priority_progress_regret_sum = 0.0
        self.priority_progress_regret_count = 0

    def add_selection(
        self,
        selected_idx: int,
        accepted_mask: np.ndarray,
        label: dict[str, np.ndarray],
        cert: dict[str, np.ndarray] | None = None,
        *,
        shortlist_mask: np.ndarray | None = None,
        fallback_used: bool = False,
    ) -> None:
        self.selected_total += 1
        self.label_metrics.add(selected_idx, label, fallback_used=fallback_used)
        valid = np.asarray(label.get("cowp/candidates/valid", []), dtype=bool)
        if valid.size == 0:
            return
        ncf = np.asarray(label.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool) & valid
        fs = np.asarray(label.get("cowp/candidates/false_safe", np.zeros_like(valid)), dtype=bool) & valid
        conv = np.asarray(label.get("cowp/candidates/conventional_safe", valid), dtype=bool) & valid
        crit = np.asarray(label.get("cowp/critical/valid", []), dtype=bool)
        witness = np.asarray(label.get("cowp/witness/exists", np.zeros((len(valid), len(crit)), dtype=bool)), dtype=bool)
        rho = np.asarray(label.get("cowp/witness/rho", np.zeros_like(witness, dtype=np.int64)), dtype=np.int64)
        if witness.ndim == 2 and crit.size and witness.shape[1] == crit.size:
            protected = ((rho == 2) | (rho == 3)) & crit[None, :]
            priority_available = protected.any(axis=1)
            priority_fs = valid & conv & (witness & protected).any(axis=1)
            priority_ncf = valid & conv & priority_available & ~priority_fs
        else:
            protected = np.zeros((len(valid), len(crit)), dtype=bool)
            priority_available = np.zeros_like(valid)
            priority_fs = np.zeros_like(valid)
            priority_ncf = np.zeros_like(valid)
        accepted = _align_candidate_vector(accepted_mask, len(valid), fill_value=False, dtype=bool) & valid
        shortlist = (
            _align_candidate_vector(shortlist_mask, len(valid), fill_value=False, dtype=bool) & valid
            if shortlist_mask is not None
            else accepted.copy()
        )
        self.shortlist_total += int(shortlist.sum())
        self.scene_any_shortlist += int(bool(shortlist.any()))
        self.scene_any_shortlist_ncf += int(bool((shortlist & ncf).any()))
        self.scene_any_valid += int(bool(valid.any()))
        self.scene_any_conventional_safe += int(bool(conv.any()))
        self.scene_any_ncf += int(bool(ncf.any()))
        self.scene_conventional_without_ncf += int(bool(conv.any() and not ncf.any()))
        self.scene_any_accepted += int(bool(accepted.any()))
        self.scene_any_accepted_ncf += int(bool((accepted & ncf).any()))
        self.scene_any_priority_ncf += int(bool(priority_ncf.any()))
        priority_eligible = valid & conv & priority_available
        self.scene_any_priority_eligible += int(bool(priority_eligible.any()))
        self.scene_priority_eligible_without_ncf += int(bool(priority_eligible.any() and not priority_ncf.any()))
        self.scene_any_accepted_priority_ncf += int(bool((accepted & priority_ncf).any()))

        # Non-coercive progress regret is conditional on the proposal bank
        # containing a protected-priority feasible candidate.  A fallback in such
        # a scene receives zero selected progress and is therefore diagnosed as
        # certificate/selector conservatism rather than proposal failure.
        if priority_ncf.any() and "cowp/candidates/trajectory" in label:
            traj_all = np.asarray(label["cowp/candidates/trajectory"])
            best_p = max((_trajectory_progress_m(traj_all[k]) for k in np.where(priority_ncf)[0]), default=0.0)
            selected_p = 0.0
            if 0 <= selected_idx < len(valid) and bool(valid[selected_idx]):
                selected_p = _trajectory_progress_m(traj_all[selected_idx])
            self.priority_progress_regret_sum += float(max(best_p - selected_p, 0.0) / max(best_p, 1.0e-6))
            self.priority_progress_regret_count += 1

        if selected_idx >= 0 and selected_idx < len(valid):
            self.selected_ncf += int(bool(ncf[selected_idx]))
            self.scene_ncf_available_selected_ncf += int(bool(ncf.any() and ncf[selected_idx]))
            self.selected_false_safe += int(bool(fs[selected_idx]))
            self.selected_conventional += int(bool(conv[selected_idx]))
            selected_priority_eligible = bool(priority_available[selected_idx] and conv[selected_idx])
            selected_priority_false_safe = bool(priority_fs[selected_idx])
            self.selected_priority_eligible += int(selected_priority_eligible)
            self.selected_priority_false_safe += int(selected_priority_false_safe)
            if fallback_used:
                self.fallback_selected_total += 1
                self.fallback_selected_with_ncf_available += int(bool(ncf.any()))
                self.fallback_selected_priority_eligible += int(selected_priority_eligible)
                self.fallback_selected_priority_false_safe += int(selected_priority_false_safe)
            tail = label.get("cowp/witness/tail_burden_excess")
            if tail is not None and witness.ndim == 2:
                tail_arr = np.asarray(tail, dtype=np.float32)
                if tail_arr.ndim == 2 and selected_idx < tail_arr.shape[0]:
                    protected_agent = protected[selected_idx] if protected.ndim == 2 else np.zeros_like(crit)
                    if protected_agent.any():
                        values = np.nan_to_num(
                            tail_arr[selected_idx][protected_agent],
                            nan=0.0,
                            posinf=2.0,
                            neginf=0.0,
                        )
                        # The selected plan is only as non-coercive as its worst
                        # protected relation.  Store that candidate-level value;
                        # finish() then aggregates the worst quartile of scenes.
                        self.protected_bte_values.append(float(np.max(values)))
            if cert is not None:
                try:
                    self.cert_selected_count += 1
                    self.cert_selected_ncf_sum += float(np.asarray(cert.get("ncf_prob"))[selected_idx])
                    self.cert_selected_fs_sum += float(np.asarray(cert.get("false_safe_prob"))[selected_idx])
                    self.cert_selected_quality_sum += float(np.asarray(cert.get("quality_prob"))[selected_idx])
                    self.cert_selected_risk_sum += float(np.asarray(cert.get("risk"))[selected_idx])
                    if cert.get("pressure_prior") is not None:
                        self.cert_selected_pressure_sum += float(np.asarray(cert.get("pressure_prior"))[selected_idx])
                except Exception:
                    pass
        if cert is not None and accepted.any():
            try:
                risk = _align_candidate_vector(cert.get("risk"), len(valid), fill_value=0.0, dtype=np.float32)
                self.cert_accepted_count += int(accepted.sum())
                self.cert_accepted_risk_sum += float(risk[accepted].sum())
                if cert.get("pressure_prior") is not None:
                    pressure = _align_candidate_vector(cert.get("pressure_prior"), len(valid), fill_value=0.0, dtype=np.float32)
                    self.cert_accepted_pressure_sum += float(pressure[accepted].sum())
            except Exception:
                pass
        self.accepted_total += int(accepted.sum())
        self.valid_total += int(valid.sum())
        self.accepted_ncf += int((accepted & ncf).sum())
        self.total_ncf += int(ncf.sum())
        self.accepted_false_safe += int((accepted & fs).sum())
        self.total_false_safe += int(fs.sum())
        self.accepted_priority_total += int((accepted & priority_available & conv).sum())
        self.accepted_priority_ncf += int((accepted & priority_ncf).sum())
        self.total_priority_ncf += int(priority_ncf.sum())
        self.accepted_priority_false_safe += int((accepted & priority_fs).sum())
        self.total_priority_false_safe += int(priority_fs.sum())
        rollout_valid = label.get("waymax/candidate_rollout_valid")
        if selected_idx >= 0 and rollout_valid is not None:
            rv = np.asarray(rollout_valid, dtype=bool)
            if selected_idx < len(rv) and bool(rv[selected_idx]):
                self.selected_waymax_valid += 1
                collision = np.asarray(label.get("waymax/candidate_collision", np.zeros_like(rv)), dtype=bool)
                offroad = np.asarray(label.get("waymax/candidate_offroad", np.zeros_like(rv)), dtype=bool)
                logdiv = np.asarray(label.get("waymax/candidate_log_divergence", np.zeros_like(rv, dtype=np.float32)), dtype=np.float32)
                self.selected_waymax_collision += int(selected_idx < len(collision) and bool(collision[selected_idx]))
                self.selected_waymax_offroad += int(selected_idx < len(offroad) and bool(offroad[selected_idx]))
                if selected_idx < len(logdiv) and np.isfinite(logdiv[selected_idx]):
                    self.selected_waymax_logdiv_sum += float(logdiv[selected_idx])
                    self.selected_waymax_logdiv_count += 1

    def add_witness_quality(self, row: dict[str, float]) -> None:
        self.witness_count += 1
        for k, v in row.items():
            self.witness_sums[k] = self.witness_sums.get(k, 0.0) + float(v)

    def finish(self, *, auprc: float, rank_good: int, rank_total: int, witness_threshold: float) -> dict[str, object]:
        metrics: dict[str, object] = self.label_metrics.finish()
        # Version the evaluation semantics so calibration cannot silently reuse
        # pre-v16.8.2 rows where the Pareto shortlist was mislabeled as the
        # certificate-accepted set and explicit valid-index fallbacks were lost.
        metrics["CertificateSemantics/Version"] = "v16_8_2_decoupled"
        metrics["FallbackSemantics/ExplicitAccounting"] = True
        if self.witness_count:
            for k, v in self.witness_sums.items():
                metrics[f"WitnessQuality/{k}"] = float(v / max(self.witness_count, 1))
        metrics["SelectedNCFRate"] = float(self.selected_ncf / max(self.selected_total, 1))
        metrics["SelectedFalseSafeRate"] = float(self.selected_false_safe / max(self.selected_total, 1))
        metrics["SelectedConventionalSafeRate"] = float(self.selected_conventional / max(self.selected_total, 1))
        metrics["PriorityBurdenTransferRate"] = float(
            self.selected_priority_false_safe / max(self.selected_priority_eligible, 1)
        )
        # This is the semantic certificate rate, not the Pareto shortlist rate.
        metrics["LearnedAcceptedCandidateRate"] = float(self.accepted_total / max(self.valid_total, 1))
        metrics["SelectionShortlist/CandidateRate"] = float(self.shortlist_total / max(self.valid_total, 1))
        metrics["SelectionShortlist/AnySceneRate"] = float(self.scene_any_shortlist / max(self.selected_total, 1))
        metrics["SelectionShortlist/AnyNCFSceneRate"] = float(self.scene_any_shortlist_ncf / max(self.selected_total, 1))
        metrics["FallbackSelection/SelectedCandidateRate"] = float(self.fallback_selected_total / max(self.selected_total, 1))
        metrics["FallbackSelection/PriorityBurdenTransferRate"] = float(
            self.fallback_selected_priority_false_safe / max(self.fallback_selected_priority_eligible, 1)
        )
        metrics["LearnedAcceptNCFRecall"] = float(self.accepted_ncf / max(self.total_ncf, 1))
        metrics["LearnedAcceptNCFPrecision"] = float(self.accepted_ncf / max(self.accepted_total, 1))
        metrics["LearnedAcceptFalseSafeRate"] = float(self.accepted_false_safe / max(self.total_false_safe, 1))
        metrics["PriorityCertificate/AcceptNCFRecall"] = float(
            self.accepted_priority_ncf / max(self.total_priority_ncf, 1)
        )
        metrics["PriorityCertificate/AcceptNCFPrecision"] = float(
            self.accepted_priority_ncf / max(self.accepted_priority_total, 1)
        )
        metrics["PriorityCertificate/AcceptFalseSafeRate"] = float(
            self.accepted_priority_false_safe / max(self.total_priority_false_safe, 1)
        )
        metrics["ProposalCoverage/AnyValidSceneRate"] = float(self.scene_any_valid / max(self.selected_total, 1))
        metrics["ProposalCoverage/AnyConventionalSafeSceneRate"] = float(self.scene_any_conventional_safe / max(self.selected_total, 1))
        metrics["ProposalCoverage/AnyNCFSceneRate"] = float(self.scene_any_ncf / max(self.selected_total, 1))
        proposal_fs_floor = float(self.scene_conventional_without_ncf / max(self.selected_total, 1))
        metrics["ProposalCoverage/ConventionalWithoutNCFSceneRate"] = proposal_fs_floor
        metrics["ProposalCoverage/BestCaseSelectedFalseSafeLowerBound"] = proposal_fs_floor
        metrics["ProposalCoverage/AnyPriorityEligibleSceneRate"] = float(
            self.scene_any_priority_eligible / max(self.selected_total, 1)
        )
        metrics["ProposalCoverage/AnyPriorityNCFSceneRate"] = float(
            self.scene_any_priority_ncf / max(self.selected_total, 1)
        )
        proposal_pbtr_floor = float(
            self.scene_priority_eligible_without_ncf / max(self.scene_any_priority_eligible, 1)
        )
        metrics["ProposalCoverage/PriorityEligibleWithoutNCFSceneRate"] = float(
            self.scene_priority_eligible_without_ncf / max(self.selected_total, 1)
        )
        metrics["ProposalCoverage/BestCasePBTRLowerBound"] = proposal_pbtr_floor
        metrics["Selector/NCFSelectionRecallGivenAvailable"] = float(
            self.scene_ncf_available_selected_ncf / max(self.scene_any_ncf, 1)
        )
        metrics["Selector/FalseSafeExcessAboveProposalFloor"] = float(
            metrics["SelectedFalseSafeRate"] - proposal_fs_floor
        )
        metrics["FallbackSelection/AnyNCFCandidateSceneRate"] = float(
            self.fallback_selected_with_ncf_available / max(self.fallback_selected_total, 1)
        )
        metrics["ProposalDiagnostics/Version"] = "v16_8_4_boundary_consistent_proposal_floor"
        metrics["CertificateCoverage/AnyAcceptedSceneRate"] = float(self.scene_any_accepted / max(self.selected_total, 1))
        metrics["CertificateCoverage/AnyAcceptedNCFSceneRate"] = float(self.scene_any_accepted_ncf / max(self.selected_total, 1))
        metrics["CertificateCoverage/EmptySceneRate"] = float(1.0 - self.scene_any_accepted / max(self.selected_total, 1))
        metrics["CertificateCoverage/NCFSceneRetention"] = float(self.scene_any_accepted_ncf / max(self.scene_any_ncf, 1))
        metrics["PriorityCertificate/NCFSceneRetention"] = float(
            self.scene_any_accepted_priority_ncf / max(self.scene_any_priority_ncf, 1)
        )
        metrics["PriorityCertificate/NonCoerciveProgressRegret"] = float(
            self.priority_progress_regret_sum / max(self.priority_progress_regret_count, 1)
        )
        if self.protected_bte_values:
            protected_bte = np.sort(np.asarray(self.protected_bte_values, dtype=np.float64))
            tail_count = max(1, int(np.ceil(0.25 * protected_bte.size)))
            metrics["PriorityBurden/MeanWorstRelationBTE"] = float(protected_bte.mean())
            metrics["PriorityBurden/BTE_CVaR_25"] = float(protected_bte[-tail_count:].mean())
            metrics["PriorityBurden/BTE_CVaR_25_Count"] = int(tail_count)
        else:
            metrics["PriorityBurden/MeanWorstRelationBTE"] = 0.0
            metrics["PriorityBurden/BTE_CVaR_25"] = 0.0
            metrics["PriorityBurden/BTE_CVaR_25_Count"] = 0
        metrics["WitnessQuality/AUPRC"] = float(auprc)
        metrics["PlannerRankingPairAccuracy"] = float(rank_good / max(rank_total, 1)) if rank_total else 0.0
        if self.selected_waymax_valid > 0:
            metrics["SelectedWaymaxRolloutValid"] = int(self.selected_waymax_valid)
            metrics["SelectedWaymaxCollisionRate"] = float(self.selected_waymax_collision / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxOffroadRate"] = float(self.selected_waymax_offroad / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxUnsafeRate"] = float((self.selected_waymax_collision + self.selected_waymax_offroad) / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxOutcomeCoverage"] = float(self.selected_waymax_valid / max(self.selected_total, 1))
            metrics["SelectedWaymaxFiniteLogDivergenceCount"] = int(self.selected_waymax_logdiv_count)
            metrics["SelectedWaymaxMeanLogDivergence"] = (
                float(self.selected_waymax_logdiv_sum / self.selected_waymax_logdiv_count)
                if self.selected_waymax_logdiv_count > 0
                else None
            )
        if self.cert_selected_count > 0:
            metrics["CandidateCertificate/SelectedNcfProbMean"] = float(self.cert_selected_ncf_sum / max(self.cert_selected_count, 1))
            metrics["CandidateCertificate/SelectedFalseSafeProbMean"] = float(self.cert_selected_fs_sum / max(self.cert_selected_count, 1))
            metrics["CandidateCertificate/SelectedQualityProbMean"] = float(self.cert_selected_quality_sum / max(self.cert_selected_count, 1))
            metrics["CandidateCertificate/SelectedRiskMean"] = float(self.cert_selected_risk_sum / max(self.cert_selected_count, 1))
            metrics["CandidateCertificate/SelectedPressurePriorMean"] = float(self.cert_selected_pressure_sum / max(self.cert_selected_count, 1))
        if self.cert_accepted_count > 0:
            metrics["CandidateCertificate/AcceptedRiskMean"] = float(self.cert_accepted_risk_sum / max(self.cert_accepted_count, 1))
            metrics["CandidateCertificate/AcceptedPressurePriorMean"] = float(self.cert_accepted_pressure_sum / max(self.cert_accepted_count, 1))
        metrics["num_scenes"] = int(self.selected_total)
        metrics["mode"] = "learned_offline"
        metrics["witness_threshold"] = float(witness_threshold)
        return metrics


def _learned_offline_candidate_eval_many(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    witness_thresholds: list[float] | tuple[float, ...] = (0.5,),
    bcot_risk_budget: float | None = None,
    bcot_risk_budgets: list[float] | tuple[float, ...] | None = None,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    methods: list[str] | tuple[str, ...] | None = None,
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    subset_modulo: int = 1,
    subset_remainder: int = 0,
    max_scenes: int | None = None,
) -> dict | dict[str, dict]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from cowp.data.dataset import TorchCOWPDataset, collate_torch
    configure_dataloader_runtime()
    from cowp.models.cowp_model import COWPModel

    thresholds = sorted({float(t) for t in witness_thresholds})
    if not thresholds:
        thresholds = [0.5]
    configured_budget = float(
        (cfg or {}).get("planning", {}).get("candidate_transport_budget", 0.35)
    )
    if bcot_risk_budgets is not None:
        budgets = sorted({float(x) for x in bcot_risk_budgets})
    else:
        budgets = [configured_budget if bcot_risk_budget is None else float(bcot_risk_budget)]
    if not budgets:
        budgets = [configured_budget]
    operating_points = [(th, budget) for th in thresholds for budget in budgets]
    multi_budget = len(budgets) > 1
    multi_method = methods is not None
    method_list = list(dict.fromkeys(str(x).strip() for x in (methods or [method]) if str(x).strip()))
    if not method_list:
        method_list = [str(method or "cowp")]
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ckpt = torch.load(checkpoint, map_location=dev)
    model_cfg = ckpt.get("cfg", cfg)
    model = COWPModel(model_cfg).to(dev)
    _load_state_dict_compatible(model, ckpt["model"])
    model.eval()
    del ckpt

    # Use the stage-filtered evaluation view.  This avoids reading dense response
    # targets and broad waymax/* tensors that are irrelevant for planner eval.
    base_ds = TorchCOWPDataset(
        cache_dir, stage="planner_eval",
        include_response_traj=False, include_response_components=False,
    )
    modulo = max(int(subset_modulo), 1)
    remainder = int(subset_remainder)
    if not 0 <= remainder < modulo:
        raise ValueError(f"subset_remainder must be in [0, subset_modulo), got {remainder}/{modulo}")
    subset_indices = [i for i in range(len(base_ds)) if i % modulo == remainder]
    if max_scenes is not None and int(max_scenes) > 0:
        subset_indices = subset_indices[: int(max_scenes)]
    if not subset_indices:
        raise ValueError(
            f"learned-offline subset is empty: size={len(base_ds)} modulo={modulo} remainder={remainder}"
        )
    ds = base_ds if modulo == 1 and len(subset_indices) == len(base_ds) else Subset(base_ds, subset_indices)
    subset_signature = __import__("hashlib").sha256(
        np.asarray(subset_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    workers = max(int(num_workers), 0)
    use_pin_memory = bool(dev.type == "cuda") if pin_memory is None else bool(pin_memory)
    dl_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_torch,
        "pin_memory": use_pin_memory,
    }
    if workers > 0:
        dl_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
        dl_kwargs["persistent_workers"] = True
    dl = DataLoader(ds, **dl_kwargs)
    beta_default = float(cfg.get("burden", {}).get("beta0_vehicle", 0.65))
    accs = {
        method_name: {
            op: _LearnedMetricsAccumulator(beta_default=beta_default)
            for op in operating_points
        }
        for method_name in method_list
    }
    pair_scores: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    cert_ncf_scores: list[np.ndarray] = []
    cert_ncf_targets: list[np.ndarray] = []
    cert_fs_scores: list[np.ndarray] = []
    cert_fs_targets: list[np.ndarray] = []
    cert_q_scores: list[np.ndarray] = []
    cert_q_targets: list[np.ndarray] = []
    priority_transport_scores: list[np.ndarray] = []
    priority_transport_targets: list[np.ndarray] = []
    global_transport_scores: list[np.ndarray] = []
    global_transport_targets: list[np.ndarray] = []
    root_direct_scores: list[np.ndarray] = []
    root_direct_targets: list[np.ndarray] = []
    root_conflict_scores: list[np.ndarray] = []
    root_conflict_targets: list[np.ndarray] = []
    root_priority_conflict_scores: list[np.ndarray] = []
    root_priority_conflict_targets: list[np.ndarray] = []
    root_aux_conflict_scores: list[np.ndarray] = []
    root_aux_conflict_targets: list[np.ndarray] = []
    root_assignment_ade_sum = 0.0
    root_assignment_ade_count = 0
    priority_transport_rank_good = 0
    priority_transport_rank_total = 0
    global_transport_rank_good = 0
    global_transport_rank_total = 0
    cert_rank_good = 0
    cert_rank_total = 0
    rank_good = 0
    rank_total = 0
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc="Learned offline COWP eval", unit="batch")
    with torch.inference_mode():
        for batch in iterator:
            batch = {
                k: v.to(dev, non_blocking=use_pin_memory)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            if not batch:
                continue
            pred = model(batch, stage="planner")
            if "critical_mask" in pred:
                batch = dict(batch)
                batch["cowp/critical/valid"] = pred["critical_mask"].bool()
            # Training and mechanism evaluation must use one canonical definition.
            # v16.7 trained on reconstructed paper-aligned targets but evaluated
            # against stale v9 false-safe/NCF labels, making calibration impossible
            # to interpret.  Rebuild the labels here from the same transport fields.
            ncf_cfg = cfg.get("ncf", {}) if isinstance(cfg, dict) else {}
            eval_target_weights = {
                "paper_aligned_witness_targets": 1.0,
                "set_transport_probability_floor": float(ncf_cfg.get("root_probability_floor", 0.02)),
                "set_transport_min_alt_weight": float(ncf_cfg.get("min_alt_weight", 0.03)),
                "set_transport_cvar_tail_mass": float(ncf_cfg.get("cvar_tail_mass", 0.25)),
                "witness_conflict_mass_floor": float(ncf_cfg.get("positive_min_natural_conflict_mass", 0.10)),
                "witness_burden_gamma": float(ncf_cfg.get("gamma", 0.10)),
                "witness_opr_alpha": float(ncf_cfg.get("alpha_opr", 0.35)),
            }
            batch = paper_aligned_supervision_batch(batch, eval_target_weights)
            alpha = float(cfg.get("planning", {}).get("alpha_opr_infer", cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            B = int(batch["cowp/candidates/valid"].shape[0])
            batch_labels = [_slim_label_from_batch_item(batch, i) for i in range(B)]
            cand_mask = batch["cowp/candidates/valid"].bool()
            crit_mask = batch["cowp/critical/valid"].bool()
            pair_mask_t = cand_mask[:, :, None] & crit_mask[:, None, :]
            pair_mask = pair_mask_t.detach().cpu().numpy()
            prob_t, _, _ = _witness_probability_and_certificate(pred["witness"], cfg)
            pair_score_np = prob_t.detach().cpu().numpy()
            gt_exists = batch["cowp/witness/exists"].bool().detach().cpu().numpy()
            if pair_mask.any():
                pair_scores.append(pair_score_np[pair_mask])
                pair_targets.append(gt_exists[pair_mask])
            pred_token = pred["witness"]["token_logits"].argmax(dim=-1).detach().cpu().numpy()
            pred_interval = pred["witness"]["conflict_interval"].round().detach().cpu().numpy()
            gt_token = batch["cowp/witness/token"].long().detach().cpu().numpy()
            gt_interval = batch["cowp/witness/conflict_interval"].detach().cpu().numpy()

            score_np = pred["planner_score"].detach().cpu().numpy()
            ncf_np = batch["cowp/candidates/noncoercive_feasible"].bool().detach().cpu().numpy()
            fs_np = batch["cowp/candidates/false_safe"].bool().detach().cpu().numpy()
            valid_np = cand_mask.detach().cpu().numpy()
            cert_ncf_t, cert_fs_t, cert_q_t = _candidate_certificate_scores(pred, pred["planner_score"], cfg, cand_mask)
            cert_risk_t = _candidate_certificate_risk(cert_ncf_t, cert_fs_t, cert_q_t, cfg)
            pressure_prior_t = _candidate_pressure_prior_torch(batch, cfg, pred["planner_score"].detach().float())
            rule_risk_t = batch.get("cowp/candidates/rule_risk")
            if rule_risk_t is None:
                rule_risk_t = torch.zeros_like(pred["planner_score"].detach().float())
            cert_ncf_np = cert_ncf_t.detach().cpu().numpy()
            cert_fs_np = cert_fs_t.detach().cpu().numpy()
            cert_q_np = cert_q_t.detach().cpu().numpy()
            cert_risk_np = cert_risk_t.detach().cpu().numpy()
            transport_t = pred.get("candidate_transport_risk")
            if torch.is_tensor(transport_t):
                transport_np = torch.nan_to_num(
                    transport_t.detach().float(), nan=1.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0).cpu().numpy()
            else:
                transport_np = cert_risk_np
            global_transport_t = pred.get("candidate_global_transport_risk")
            if torch.is_tensor(global_transport_t):
                global_transport_np = torch.nan_to_num(
                    global_transport_t.detach().float(), nan=1.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0).cpu().numpy()
            else:
                global_transport_np = transport_np
            rho_t = batch.get("cowp/witness/rho")
            if torch.is_tensor(rho_t):
                protected_t = ((rho_t.long() == 2) | (rho_t.long() == 3)) & crit_mask[:, None, :]
                protected_available_t = protected_t.any(dim=-1)
                priority_fs_t = cand_mask & batch.get("cowp/candidates/conventional_safe", cand_mask).bool() & (
                    batch["cowp/witness/exists"].bool() & protected_t
                ).any(dim=-1)
                priority_ncf_t = cand_mask & batch.get("cowp/candidates/conventional_safe", cand_mask).bool() & protected_available_t & ~priority_fs_t
            else:
                priority_fs_t = torch.zeros_like(cand_mask)
                priority_ncf_t = torch.zeros_like(cand_mask)
            priority_fs_np = priority_fs_t.detach().cpu().numpy()
            priority_ncf_np = priority_ncf_t.detach().cpu().numpy()
            pressure_prior_np = pressure_prior_t.detach().cpu().numpy()
            rule_risk_np = rule_risk_t.detach().cpu().numpy()
            if valid_np.any():
                cert_ncf_scores.append(cert_ncf_np[valid_np])
                cert_ncf_targets.append(ncf_np[valid_np])
                cert_fs_scores.append(cert_fs_np[valid_np])
                cert_fs_targets.append(fs_np[valid_np])
                cert_q_scores.append(cert_q_np[valid_np])
                cert_q_targets.append(ncf_np[valid_np] & ~fs_np[valid_np])
                priority_disc_np = valid_np & (priority_fs_np | priority_ncf_np)
                if priority_disc_np.any():
                    priority_transport_scores.append(transport_np[priority_disc_np])
                    priority_transport_targets.append(priority_fs_np[priority_disc_np])
                global_disc_np = valid_np & (fs_np | ncf_np)
                if global_disc_np.any():
                    global_transport_scores.append(global_transport_np[global_disc_np])
                    global_transport_targets.append(fs_np[global_disc_np])

            root_eval = _root_transport_eval_arrays(pred, batch)
            if root_eval is not None:
                all_root = root_eval["all_mask"]
                conflict_root = root_eval["conflict_mask"]
                priority_conflict_root = root_eval["priority_conflict_mask"]
                if all_root.any():
                    root_direct_scores.append(root_eval["direct"][all_root])
                    root_direct_targets.append(root_eval["target"][all_root])
                if conflict_root.any():
                    root_conflict_scores.append(root_eval["direct"][conflict_root])
                    root_conflict_targets.append(root_eval["target"][conflict_root])
                    if root_eval["aux"] is not None:
                        root_aux_conflict_scores.append(root_eval["aux"][conflict_root])
                        root_aux_conflict_targets.append(root_eval["target"][conflict_root])
                if priority_conflict_root.any():
                    root_priority_conflict_scores.append(root_eval["direct"][priority_conflict_root])
                    root_priority_conflict_targets.append(root_eval["target"][priority_conflict_root])
                root_assignment_ade_sum += float(root_eval["assignment_ade_sum"])
                root_assignment_ade_count += int(root_eval["assignment_ade_count"])
            g, t = _ranking_pair_accuracy(score_np, ncf_np, fs_np, valid_np)
            rank_good += g
            rank_total += t
            # _ranking_pair_accuracy expects lower score for first positive class;
            # here lower risk should rank NCF before false-safe, so pass ncf as pos.
            cg, ct = _ranking_pair_accuracy(cert_risk_np, ncf_np, fs_np, valid_np)
            cert_rank_good += cg
            cert_rank_total += ct
            ptg, ptt = _ranking_pair_accuracy(transport_np, priority_ncf_np, priority_fs_np, valid_np)
            priority_transport_rank_good += ptg
            priority_transport_rank_total += ptt
            gtg, gtt = _ranking_pair_accuracy(global_transport_np, ncf_np, fs_np, valid_np)
            global_transport_rank_good += gtg
            global_transport_rank_total += gtt

            # Model inference and all host transfers above are shared across
            # methods.  Only the inexpensive candidate selection/aggregation is
            # repeated, replacing N full checkpoint loads and N cache scans with
            # one pass over the validation set.
            for method_name in method_list:
                for th, budget in operating_points:
                    batch_selected, batch_accepted_masks, batch_shortlist_masks, batch_fallback_flags = _select_from_learned(
                        batch,
                        pred,
                        witness_threshold=th,
                        bcot_risk_budget=budget,
                        alpha_opr=alpha,
                        gate_mode=gate_mode,
                        secondary_witness_threshold=secondary_witness_threshold,
                        secondary_opr_alpha=secondary_opr_alpha,
                        priority_hard_threshold=priority_hard_threshold,
                        soft_ncf_penalty=soft_ncf_penalty,
                        method=method_name,
                        offline_fallback=offline_fallback,
                        adaptive_frontier_margin=adaptive_frontier_margin,
                        outcome_risk_penalty=outcome_risk_penalty,
                        outcome_risk_threshold=outcome_risk_threshold,
                        cfg=cfg,
                    )
                    pred_exists = pair_score_np >= float(th)
                    acc = accs[method_name][(th, budget)]
                    for i, item in enumerate(batch_labels):
                        cert_item = {
                            "ncf_prob": cert_ncf_np[i],
                            "false_safe_prob": cert_fs_np[i],
                            "quality_prob": cert_q_np[i],
                            "risk": cert_risk_np[i],
                            "pressure_prior": pressure_prior_np[i],
                            "rule_risk": rule_risk_np[i],
                        }
                        acc.add_selection(
                            int(batch_selected[i]),
                            np.asarray(batch_accepted_masks[i], dtype=bool),
                            item,
                            cert=cert_item,
                            shortlist_mask=np.asarray(batch_shortlist_masks[i], dtype=bool),
                            fallback_used=bool(batch_fallback_flags[i]),
                        )
                        acc.add_witness_quality(witness_quality(pred_exists[i], pred_token[i], pred_interval[i], gt_exists[i], gt_token[i], gt_interval[i], pair_mask[i]))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    done=accs[method_list[0]][operating_points[0]].selected_total,
                    methods=len(method_list),
                    operating_points=len(operating_points),
                    refresh=True,
                )

    all_pair_scores = np.concatenate(pair_scores) if pair_scores else np.asarray([], dtype=np.float32)
    auprc = _average_precision_binary(all_pair_scores, np.concatenate(pair_targets)) if pair_scores else 0.0
    cert_ncf_auprc = _average_precision_binary(np.concatenate(cert_ncf_scores), np.concatenate(cert_ncf_targets)) if cert_ncf_scores else 0.0
    cert_fs_auprc = _average_precision_binary(np.concatenate(cert_fs_scores), np.concatenate(cert_fs_targets)) if cert_fs_scores else 0.0
    cert_q_auprc = _average_precision_binary(np.concatenate(cert_q_scores), np.concatenate(cert_q_targets)) if cert_q_scores else 0.0
    priority_transport_fs_auprc = _average_precision_binary(
        np.concatenate(priority_transport_scores), np.concatenate(priority_transport_targets)
    ) if priority_transport_scores else 0.0
    global_transport_fs_auprc = _average_precision_binary(
        np.concatenate(global_transport_scores), np.concatenate(global_transport_targets)
    ) if global_transport_scores else 0.0
    root_all_score = np.concatenate(root_direct_scores) if root_direct_scores else np.asarray([], dtype=np.float32)
    root_all_target = np.concatenate(root_direct_targets) if root_direct_targets else np.asarray([], dtype=bool)
    root_conflict_score = np.concatenate(root_conflict_scores) if root_conflict_scores else np.asarray([], dtype=np.float32)
    root_conflict_target = np.concatenate(root_conflict_targets) if root_conflict_targets else np.asarray([], dtype=bool)
    root_aux_conflict_score = np.concatenate(root_aux_conflict_scores) if root_aux_conflict_scores else np.asarray([], dtype=np.float32)
    root_aux_conflict_target = np.concatenate(root_aux_conflict_targets) if root_aux_conflict_targets else np.asarray([], dtype=bool)
    root_all_auprc = _average_precision_binary(root_all_score, root_all_target)
    root_conflict_auprc = _average_precision_binary(root_conflict_score, root_conflict_target)
    root_conflict_recall = _binary_recall_at(root_conflict_score, root_conflict_target, 0.5)
    root_priority_conflict_score = np.concatenate(root_priority_conflict_scores) if root_priority_conflict_scores else np.asarray([], dtype=np.float32)
    root_priority_conflict_target = np.concatenate(root_priority_conflict_targets) if root_priority_conflict_targets else np.asarray([], dtype=bool)
    root_priority_conflict_auprc = _average_precision_binary(root_priority_conflict_score, root_priority_conflict_target)
    root_priority_conflict_recall = _binary_recall_at(root_priority_conflict_score, root_priority_conflict_target, 0.5)
    root_aux_conflict_auprc = _average_precision_binary(root_aux_conflict_score, root_aux_conflict_target)
    root_assignment_minade = root_assignment_ade_sum / max(root_assignment_ade_count, 1)
    out_by_method = {
        method_name: {
            op: accs[method_name][op].finish(
                auprc=auprc,
                rank_good=rank_good,
                rank_total=rank_total,
                witness_threshold=op[0],
            )
            for op in operating_points
        }
        for method_name in method_list
    }
    for method_out in out_by_method.values():
        for (th, budget), row in method_out.items():
            row["CandidateCertificate/NCF_AUPRC"] = float(cert_ncf_auprc)
            row["CandidateCertificate/FalseSafe_AUPRC"] = float(cert_fs_auprc)
            row["CandidateCertificate/Quality_AUPRC"] = float(cert_q_auprc)
            row["CandidateCertificate/RiskRankingPairAccuracy"] = float(cert_rank_good / max(cert_rank_total, 1)) if cert_rank_total else 0.0
            row["BCOT/PriorityFalseSafe_AUPRC"] = float(priority_transport_fs_auprc)
            row["BCOT/GlobalFalseSafe_AUPRC"] = float(global_transport_fs_auprc)
            # Backward-compatible alias now follows the decision certificate,
            # which is priority-aware; global burden transfer is reported above.
            row["BCOT/FalseSafe_AUPRC"] = float(priority_transport_fs_auprc)
            # Direct mechanism metrics: does the root-indexed head recover an
            # explicit low-burden safe response for each natural option?
            row["RootTransport/LowSafeExist_AUPRC"] = float(root_all_auprc)
            row["RootTransport/ConflictConditioned_AUPRC"] = float(root_conflict_auprc)
            row["RootTransport/ConflictConditioned_Recall@0.5"] = float(root_conflict_recall)
            row["RootTransport/PriorityConflict_AUPRC"] = float(root_priority_conflict_auprc)
            row["RootTransport/PriorityConflict_Recall@0.5"] = float(root_priority_conflict_recall)
            row["RootTransport/AuxConflictConditioned_AUPRC"] = float(root_aux_conflict_auprc)
            row["RootTransport/NaturalAssignmentMinADE_m"] = float(root_assignment_minade)
            row["RootTransport/EvaluatedConflictRoots"] = int(root_conflict_score.size)
            row["bcot_risk_budget"] = float(budget)
            row["pair_witness_threshold"] = float(th)
            row["BCOT/PriorityRiskRankingPairAccuracy"] = float(
                priority_transport_rank_good / max(priority_transport_rank_total, 1)
            ) if priority_transport_rank_total else 0.0
            row["BCOT/GlobalRiskRankingPairAccuracy"] = float(
                global_transport_rank_good / max(global_transport_rank_total, 1)
            ) if global_transport_rank_total else 0.0
            row["BCOT/RiskRankingPairAccuracy"] = row["BCOT/PriorityRiskRankingPairAccuracy"]
            row["EvaluationSubset/Modulo"] = int(modulo)
            row["EvaluationSubset/Remainder"] = int(remainder)
            row["EvaluationSubset/Scenes"] = int(len(subset_indices))
            row["EvaluationSubset/IndexSHA256"] = subset_signature
    if all_pair_scores.size:
        qs = np.quantile(all_pair_scores, [0.1, 0.5, 0.9, 0.99])
        for method_out in out_by_method.values():
            for row in method_out.values():
                row["WitnessProb/p10"] = float(qs[0])
                row["WitnessProb/p50"] = float(qs[1])
                row["WitnessProb/p90"] = float(qs[2])
                row["WitnessProb/p99"] = float(qs[3])
    if not multi_budget:
        # Backward-compatible public shape: dict[witness_threshold, metrics].
        simple = {
            method_name: {op[0]: row for op, row in method_out.items()}
            for method_name, method_out in out_by_method.items()
        }
        return simple if multi_method else simple[method_list[0]]
    return out_by_method if multi_method else out_by_method[method_list[0]]


def learned_offline_candidate_eval(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    witness_threshold: float = 0.5,
    bcot_risk_budget: float | None = None,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    subset_modulo: int = 1,
    subset_remainder: int = 0,
    max_scenes: int | None = None,
) -> dict[str, object]:
    return _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        witness_thresholds=[float(witness_threshold)],
        bcot_risk_budget=bcot_risk_budget,
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
        subset_modulo=subset_modulo,
        subset_remainder=subset_remainder,
        max_scenes=max_scenes,
    )[float(witness_threshold)]


def learned_offline_candidate_eval_sweep(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    witness_thresholds: list[float] | tuple[float, ...] = (0.5,),
    bcot_risk_budget: float | None = None,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    subset_modulo: int = 1,
    subset_remainder: int = 0,
    max_scenes: int | None = None,
) -> list[dict[str, object]]:
    out = _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        witness_thresholds=list(witness_thresholds),
        bcot_risk_budget=bcot_risk_budget,
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
        subset_modulo=subset_modulo,
        subset_remainder=subset_remainder,
        max_scenes=max_scenes,
    )
    return [out[th] for th in sorted(out)]


def learned_offline_candidate_eval_budget_sweep(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    bcot_risk_budgets: list[float] | tuple[float, ...],
    witness_threshold: float = 0.70,
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    subset_modulo: int = 1,
    subset_remainder: int = 0,
    max_scenes: int | None = None,
) -> list[dict[str, object]]:
    """Sweep the candidate BCOT budget from one shared model/cache pass.

    Pair-witness probability and candidate transported-option risk are different
    random variables.  v11 swept one scalar while reusing it for both, making the
    selected operating point statistically uninterpretable.  This API keeps the
    pair threshold fixed and sweeps only the candidate feasibility budget.
    """
    nested = _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        witness_thresholds=[float(witness_threshold)],
        bcot_risk_budgets=list(bcot_risk_budgets),
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
        subset_modulo=subset_modulo,
        subset_remainder=subset_remainder,
        max_scenes=max_scenes,
    )
    rows = [row for (_, _), row in sorted(nested.items(), key=lambda kv: kv[0][1])]
    return rows


def learned_offline_candidate_eval_methods(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    methods: list[str] | tuple[str, ...],
    batch_size: int = 8,
    device: str = "auto",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    witness_thresholds: list[float] | tuple[float, ...] = (0.5,),
    bcot_risk_budget: float | None = None,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    subset_modulo: int = 1,
    subset_remainder: int = 0,
    max_scenes: int | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Evaluate multiple internal methods from one shared model/cache pass."""
    nested = _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        witness_thresholds=list(witness_thresholds),
        bcot_risk_budget=bcot_risk_budget,
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        methods=list(methods),
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
        subset_modulo=subset_modulo,
        subset_remainder=subset_remainder,
        max_scenes=max_scenes,
    )
    return {
        method_name: [method_out[th] for th in sorted(method_out)]
        for method_name, method_out in nested.items()
    }

def import_policy_fn(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("Policy spec must be 'module.submodule:function_name'.")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Imported policy {spec} is not callable.")
    return fn


def _call_policy(policy_fn: Callable, state, step: int, scenario_index: int):
    # Backward-compatible slow path; production rollout uses _make_policy_caller
    # below so inspect.signature is not repeated at every simulator step.
    try:
        sig = inspect.signature(policy_fn)
        kwargs = {}
        if "step" in sig.parameters:
            kwargs["step"] = step
        if "scenario_index" in sig.parameters:
            kwargs["scenario_index"] = scenario_index
        return policy_fn(state, **kwargs)
    except (TypeError, ValueError):
        return policy_fn(state)


def _make_policy_caller(policy_fn: Callable):
    """Return a fast per-step caller with the policy signature inspected once."""
    try:
        sig = inspect.signature(policy_fn)
        has_step = "step" in sig.parameters
        has_scenario = "scenario_index" in sig.parameters
    except (TypeError, ValueError):
        has_step = False
        has_scenario = False

    if has_step and has_scenario:
        def call(state, step: int, scenario_index: int):
            return policy_fn(state, step=step, scenario_index=scenario_index)
    elif has_step:
        def call(state, step: int, scenario_index: int):
            return policy_fn(state, step=step)
    elif has_scenario:
        def call(state, step: int, scenario_index: int):
            return policy_fn(state, scenario_index=scenario_index)
    else:
        def call(state, step: int, scenario_index: int):
            return policy_fn(state)
    return call


def _make_waymax_environment(max_num_objects: int | None = None, action_mode: str = "delta_xy_yaw"):
    try:
        import dataclasses
        from waymax import config as _config  # type: ignore
        from waymax import dynamics  # type: ignore
        from waymax import env as waymax_env  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.env, waymax.config and waymax.dynamics are required for real closed-loop rollout.") from exc

    # Waymax has used both names in docs/examples.  Current public Waymax
    # exposes env.BaseEnvironment; some older/local builds exposed the alias
    # MultiAgentEnvironment.  Accept both.
    env_cls = getattr(waymax_env, "MultiAgentEnvironment", None) or getattr(waymax_env, "BaseEnvironment", None)
    if env_cls is None:
        try:
            from waymax.env.base_environment import BaseEnvironment as env_cls  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unsupported Waymax API: expected waymax.env.BaseEnvironment or MultiAgentEnvironment.") from exc

    # The built-in COWP policy emits a 3-D delta action by default.  Therefore
    # DeltaGlobal is the correct dynamics for --waymax-action-mode delta_xy_yaw.
    # StateDynamics expects a 5-D absolute [x, y, yaw, vel_x, vel_y] action.
    if action_mode == "absolute_xy_yaw":
        dynamics_names = ("StateDynamics", "DeltaGlobal")
    else:
        dynamics_names = ("DeltaGlobal", "StateDynamics")
    dynamics_cls = next((getattr(dynamics, name, None) for name in dynamics_names if getattr(dynamics, name, None) is not None), None)
    if dynamics_cls is None:
        raise RuntimeError(f"Unsupported Waymax API: expected one of {dynamics_names} in waymax.dynamics.")
    dyn = dynamics_cls()

    env_config = getattr(_config, "EnvironmentConfig", lambda: None)()
    kwargs = {}
    # The built-in policy controls only the SDC/ego.  Do not set VALID here;
    # doing so freezes or overwrites every non-ego object with zero actions.
    if hasattr(_config, "ObjectType") and hasattr(_config.ObjectType, "SDC"):
        kwargs["controlled_object"] = _config.ObjectType.SDC
    # Replay scripts compute their own metric accumulator and never use rewards.
    # Keeping Waymax rewards on adds avoidable metric/reward work at every step
    # and can fail on releases where reward metrics require route observations.
    kwargs["compute_reward"] = False
    if max_num_objects is not None:
        kwargs["max_num_objects"] = int(max_num_objects)
    if env_config is not None and kwargs:
        try:
            valid_fields = {f.name for f in dataclasses.fields(env_config)} if dataclasses.is_dataclass(env_config) else set(kwargs)
            env_config = dataclasses.replace(env_config, **{k: v for k, v in kwargs.items() if k in valid_fields})
        except Exception:
            pass
    try:
        return env_cls(dynamics_model=dyn, config=env_config)
    except TypeError:
        try:
            return env_cls(dyn, env_config)
        except TypeError:
            try:
                return env_cls(dynamics_model=dyn)
            except TypeError:
                return env_cls(dyn)



def _clear_accelerator_caches() -> None:
    """Best-effort cleanup between scenarios for mixed JAX + PyTorch rollout.

    The important memory fix is not retaining SimulatorState objects in the
    rollout output.  This helper only releases Python references/caches sooner.
    """
    gc.collect()
    try:
        import jax  # type: ignore

        clear = getattr(jax, "clear_caches", None)
        if callable(clear):
            clear()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _consume_policy_diagnostics(policy_fn: Callable) -> dict | None:
    consumer = getattr(policy_fn, "consume_diagnostics", None)
    if callable(consumer):
        try:
            row = consumer()
            return row if isinstance(row, dict) else None
        except Exception:
            return None
    return None

def _env_step(env, state, action):
    result = env.step(state, action)
    if isinstance(result, tuple):
        return result[0]
    return result


class _WaymaxEnvOps:
    """Cached reset/step callables with transparent JAX-JIT fallback."""

    def __init__(self, env, *, use_jit: bool = True):
        self.env = env
        self.use_jit = bool(use_jit)
        self._reset_fn = env.reset
        self._step_fn = lambda state, action: _env_step(env, state, action)
        self._reset_jitted = False
        self._step_jitted = False
        if self.use_jit:
            try:
                import jax  # type: ignore

                self._reset_fn = jax.jit(env.reset)
                self._step_fn = jax.jit(lambda state, action: _env_step(env, state, action))
                self._reset_jitted = True
                self._step_jitted = True
            except Exception:
                self._reset_fn = env.reset
                self._step_fn = lambda state, action: _env_step(env, state, action)

    def reset(self, init_state):
        try:
            return self._reset_fn(init_state)
        except Exception:
            if not self._reset_jitted:
                raise
            self._reset_jitted = False
            self._reset_fn = self.env.reset
            return self._reset_fn(init_state)

    def step(self, state, action):
        try:
            return self._step_fn(state, action)
        except Exception:
            if not self._step_jitted:
                raise
            self._step_jitted = False
            self._step_fn = lambda current_state, current_action: _env_step(self.env, current_state, current_action)
            return self._step_fn(state, action)


def _state_done(state) -> bool:
    for attr in ("is_done", "done"):
        val = getattr(state, attr, None)
        if val is None:
            continue
        try:
            return bool(np.asarray(val).all())
        except Exception:
            try:
                return bool(val)
            except Exception:
                return False
    return False


def waymax_closed_loop_rollout(
    data_config,
    policy_fn: Callable,
    num_scenarios: int | None = None,
    horizon_steps: int | None = None,
    progress: bool = True,
    compute_standard_metrics: bool = False,
    action_mode: str = "delta_xy_yaw",
    keep_rollout_state: bool = False,
    clear_accelerator_cache: bool = False,
    split: str | None = None,
    tfexample_glob: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    reuse_env: bool = True,
    prefilter_shards: bool = True,
    jit_env: bool = True,
    jit_standard_metrics: bool = True,
    status_every: int = 10,
    standard_metric_names: set[str] | list[str] | tuple[str, ...] | None = None,
):
    """Run real Waymax closed-loop simulation by stepping a Waymax environment.

    ``policy_fn`` must return a Waymax-compatible action for the current
    SimulatorState.  This function intentionally does not use proto-derived COWP
    labels; it is the closed-loop path for validating a planner in Waymax.
    """
    from cowp.waymax_eval.dataloader import waymax_state_generator, waymax_state_generator_sharded

    horizon = int(horizon_steps) if horizon_steps is not None else 80
    num_shards = max(int(num_shards), 1)
    shard_index = int(shard_index) % num_shards
    use_prefilter = bool(prefilter_shards) and num_shards > 1
    if use_prefilter:
        gen = waymax_state_generator_sharded(
            data_config,
            split=split,
            tfexample_glob=tfexample_glob,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    else:
        gen = enumerate(waymax_state_generator(data_config, split=split, tfexample_glob=tfexample_glob))
    total = num_scenarios
    iterator = tqdm_iter(gen, enabled=progress, total=total, desc=f"Waymax closed-loop rollout shard {shard_index}/{num_shards}", unit="scenario")
    outputs = []
    env_cache: dict[tuple[int | None, str], tuple[object, _WaymaxEnvOps]] = {}
    call_policy = _make_policy_caller(policy_fn)
    started_at = time.time()
    last_status_at = started_at
    standard_metric_objects = None
    standard_metric_errors = {}
    if compute_standard_metrics:
        # Instantiate the metric objects once per rollout process.  They are used
        # as stateless compute helpers; episode aggregation state remains inside
        # each WaymaxStandardMetricAccumulator below.
        from cowp.waymax_eval.metrics_standard import build_waymax_metric_objects

        standard_metric_objects, standard_metric_errors = build_waymax_metric_objects(standard_metric_names)
    for raw_index, init_state in iterator:
        if (not use_prefilter) and num_shards > 1 and (raw_index % num_shards) != shard_index:
            continue
        scenario_index = raw_index
        max_objects = getattr(init_state, "num_objects", None)
        if max_objects is None and hasattr(init_state, "log_trajectory"):
            max_objects = getattr(init_state.log_trajectory, "num_objects", None)
        env_key = (int(max_objects) if max_objects is not None else None, str(action_mode))
        if reuse_env and env_key in env_cache:
            env, env_ops = env_cache[env_key]
        else:
            env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
            env_ops = _WaymaxEnvOps(env, use_jit=jit_env)
            if reuse_env:
                env_cache[env_key] = (env, env_ops)
        state = env_ops.reset(init_state)
        steps = 0
        policy_diagnostics = []
        metric_acc = (
            WaymaxStandardMetricAccumulator(
                metric_objects=standard_metric_objects,
                init_errors=dict(standard_metric_errors),
                jit_metrics=bool(jit_standard_metrics),
            )
            if compute_standard_metrics
            else None
        )
        for step in range(horizon):
            action = call_policy(state, step=step, scenario_index=scenario_index)
            diag = _consume_policy_diagnostics(policy_fn)
            if diag is not None:
                policy_diagnostics.append(diag)
            state = env_ops.step(state, action)
            steps += 1
            if metric_acc is not None:
                # Waymax's built-in metrics are per-current-timestep metrics.  Update
                # after every simulator step so episode CR/offroad/wrong-way are
                # any-over-rollout events, not just the final frame.
                metric_acc.update(state)
            if _state_done(state):
                break
        if metric_acc is not None and steps == 0:
            metric_acc.update(state)
        # Do not retain SimulatorState by default.  A Waymax SimulatorState can keep
        # JAX device buffers alive; storing one per scenario causes closed-loop eval
        # memory to grow with --num-scenarios.  Metrics are computed before dropping
        # the state, so the JSON payload still contains the required evaluation info.
        item = {"steps": steps, "policy_diagnostics": policy_diagnostics}
        if compute_standard_metrics and metric_acc is not None:
            item["standard_metrics"] = metric_acc.finalize()
        if keep_rollout_state:
            item["state"] = state
        outputs.append(item)
        if clear_accelerator_cache:
            _clear_accelerator_caches()
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(done=len(outputs), steps=steps, refresh=True)
        if (not progress) and int(status_every or 0) > 0:
            now = time.time()
            if len(outputs) == 1 or len(outputs) % int(status_every) == 0 or now - last_status_at >= 120.0:
                elapsed = max(now - started_at, 1e-6)
                rate = len(outputs) / elapsed
                print(
                    f"[waymax] shard={shard_index}/{num_shards} done={len(outputs)}"
                    f" raw_index={raw_index} last_steps={steps} rate={rate:.3f} scen/s elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_status_at = now
        if num_scenarios is not None and len(outputs) >= num_scenarios:
            break
    return outputs
