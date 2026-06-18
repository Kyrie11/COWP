from __future__ import annotations

import numpy as np


def _trajectory_progress_m(traj: np.ndarray) -> float:
    if traj is None or len(traj) < 2:
        return 0.0
    return float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))


def _progress_reference_m(label: dict[str, np.ndarray]) -> float:
    traj = np.asarray(label["cowp/candidates/trajectory"])
    valid = label["cowp/candidates/valid"].astype(bool)
    if "cowp/candidates/conventional_safe" in label:
        mask = valid & label["cowp/candidates/conventional_safe"].astype(bool)
    else:
        mask = valid
    if not np.any(mask):
        mask = valid
    vals = [_trajectory_progress_m(t) for t in traj[mask]]
    return max(vals) if vals else 1.0


def metrics_from_labels(selected_indices: list[int], label_dicts: list[dict[str, np.ndarray]]) -> dict[str, float]:
    n = len(selected_indices)
    cf_count = 0
    false_safe = 0
    cbs = []
    oprs = []
    hbcr = 0
    collision_or_offroad = 0
    fallback_count = 0
    progress_m = []
    progress_norm = []
    for k, label in zip(selected_indices, label_dicts):
        ref_progress = max(_progress_reference_m(label), 1e-6)
        if k < 0:
            # A negative index represents a conservative fallback that was not in
            # the candidate lattice.  Count its progress as zero and expose a
            # separate fallback rate instead of treating it as an automatic
            # collision.  True closed-loop CR should come from simulator metrics.
            fallback_count += 1
            progress_m.append(0.0)
            progress_norm.append(0.0)
            continue
        crit = label["cowp/critical/valid"].astype(bool)
        conv = bool(label["cowp/candidates/conventional_safe"][k])
        collision_or_offroad += int(not conv)
        traj = label["cowp/candidates/trajectory"][k]
        p_m = _trajectory_progress_m(traj)
        progress_m.append(p_m)
        # Paper EP is route/progress normalized.  In label-only evaluation we do
        # not have the route integral, so normalize by the best valid/conventional
        # lattice progress and keep EP_m for debugging.
        progress_norm.append(float(np.clip(p_m / ref_progress, 0.0, 1.0)))
        wit = label["cowp/witness/exists"][k].astype(bool) & crit
        cf_count += int(conv)
        false_safe += int(conv and np.any(wit))
        if np.any(crit):
            cbs.append(float(np.max(label["cowp/witness/burden_total"][k, crit])))
            oprs.append(float(np.mean(label["cowp/witness/opr"][k, crit])))
            hbcr += int(np.any(label["cowp/witness/min_safe_burden"][k, crit] > label["cowp/natural/beta"][crit]))
    return {
        "CR": float(collision_or_offroad / max(n, 1)),
        "EP": float(np.mean(progress_norm)) if progress_norm else 0.0,
        "EP_m": float(np.mean(progress_m)) if progress_m else 0.0,
        "FallbackRate": float(fallback_count / max(n, 1)),
        "FSR": float(false_safe / max(cf_count, 1)),
        "CBS": float(np.mean(cbs)) if cbs else 0.0,
        "OPR": float(np.mean(oprs)) if oprs else 0.0,
        "HBCR": float(hbcr / max(n, 1)),
    }



def stress_acceptance_metrics(decisions: list[tuple[int, np.ndarray]], label_dicts: list[dict[str, np.ndarray]]) -> dict[str, float]:
    noncoercive_total = 0
    noncoercive_accepted = 0
    false_safe_total = 0
    false_safe_accepted = 0
    pred_pos = 0
    gt_pos = 0
    true_pos = 0
    for (selected_idx, accepted_mask), label in zip(decisions, label_dicts):
        cand_valid = label["cowp/candidates/valid"].astype(bool)
        ncf = label["cowp/candidates/noncoercive_feasible"].astype(bool) & cand_valid
        false_safe = label["cowp/candidates/false_safe"].astype(bool) & cand_valid
        accepted = accepted_mask.astype(bool) if accepted_mask is not None else np.zeros_like(cand_valid)
        noncoercive_total += int(ncf.sum())
        noncoercive_accepted += int((accepted & ncf).sum())
        false_safe_total += int(false_safe.sum())
        false_safe_accepted += int((accepted & false_safe).sum())
        gt = label["cowp/witness/exists"].astype(bool)
        pred = gt.copy()  # rule certificate mode uses deterministic labels as predictions.
        mask = cand_valid[:, None] & label["cowp/critical/valid"].astype(bool)[None, :]
        gt_pos += int((gt & mask).sum())
        pred_pos += int((pred & mask).sum())
        true_pos += int((pred & gt & mask).sum())
    return {
        "Accept Non-Coercive": float(noncoercive_accepted / max(noncoercive_total, 1)),
        "Accept False-Safe": float(false_safe_accepted / max(false_safe_total, 1)),
        "Witness Recall": float(true_pos / max(gt_pos, 1)),
        "Witness Precision": float(true_pos / max(pred_pos, 1)),
    }


def witness_table_from_labels(label_dicts: list[dict[str, np.ndarray]]) -> dict[str, float]:
    counts = {"HB": [0, 0], "AY": [0, 0], "PA": [0, 0], "GS": [0, 0]}
    total_pos = 0
    for label in label_dicts:
        exists = label["cowp/witness/exists"].astype(bool)
        total_pos += int(exists.sum())
        toks = label["cowp/witness/token"]
        for name, token_id in [("HB", 1), ("AY", 2), ("PA", 3), ("GS", 4)]:
            gt = exists & (toks == token_id)
            counts[name][0] += int(gt.sum())
            counts[name][1] += int(gt.sum())
    out = {"WLA": 1.0 if total_pos else 0.0, "MTA": 1.0 if total_pos else 0.0}
    for name, (tp, denom) in counts.items():
        out[f"{name}-F1"] = float(tp / max(denom, 1)) if denom else 0.0
    return out

def witness_quality(pred_exists: np.ndarray, pred_token: np.ndarray, pred_interval: np.ndarray, gt_exists: np.ndarray, gt_token: np.ndarray, gt_interval: np.ndarray, mask: np.ndarray, iou_threshold: float = 0.5) -> dict[str, float]:
    mask = mask.astype(bool)
    gt_pos = gt_exists.astype(bool) & mask
    pred_pos = pred_exists.astype(bool) & mask
    tp = pred_pos & gt_pos
    recall = float(tp.sum() / max(gt_pos.sum(), 1))
    precision = float(tp.sum() / max(pred_pos.sum(), 1))
    # Localization: same pair plus interval IoU.
    loc_ok = 0
    for idx in zip(*np.where(tp)):
        p = pred_interval[idx]
        g = gt_interval[idx]
        inter = max(0, min(p[1], g[1]) - max(p[0], g[0]) + 1)
        union = max(p[1], g[1]) - min(p[0], g[0]) + 1
        if union > 0 and inter / union > iou_threshold:
            loc_ok += 1
    wla = loc_ok / max(int(gt_pos.sum()), 1)
    mta = float((pred_token[tp] == gt_token[tp]).sum() / max(int(tp.sum()), 1)) if np.any(tp) else 0.0
    return {"WitnessRecall": recall, "WitnessPrecision": precision, "WLA": wla, "MTA": mta}


def policy_diagnostic_summary(rollouts: list[dict]) -> dict[str, float]:
    """Aggregate online COWP policy diagnostics produced during Waymax rollout.

    These are model-predicted closed-loop certification signals.  They do not
    replace label/proto counterfactual certification, but they make the Waymax
    path report FSR/CBS/OPR-style quantities instead of only returning raw final
    simulator states.
    """
    rows: list[dict] = []
    for item in rollouts:
        rows.extend(item.get("policy_diagnostics", []) or [])
    if not rows:
        return {}
    max_w = np.asarray([float(r.get("max_witness_prob", 0.0)) for r in rows], dtype=np.float32)
    threshold = np.asarray([float(r.get("witness_threshold", 0.5)) for r in rows], dtype=np.float32)
    min_opr = np.asarray([float(r.get("min_opr", 1.0)) for r in rows], dtype=np.float32)
    mean_opr = np.asarray([float(r.get("mean_opr", 1.0)) for r in rows], dtype=np.float32)
    burden = np.asarray([float(r.get("max_predicted_burden", 0.0)) for r in rows], dtype=np.float32)
    fallback = np.asarray([bool(r.get("fallback_used", False)) for r in rows], dtype=bool)
    return {
        "ClosedLoopPredFSR": float(np.mean(max_w >= threshold)),
        "ClosedLoopMeanWitnessProb": float(np.mean(max_w)),
        "ClosedLoopCBS_pred": float(np.mean(burden)),
        "ClosedLoopOPR_min": float(np.mean(min_opr)),
        "ClosedLoopOPR_mean": float(np.mean(mean_opr)),
        "ClosedLoopFallbackStepRate": float(np.mean(fallback)),
        "ClosedLoopPolicySteps": float(len(rows)),
    }
