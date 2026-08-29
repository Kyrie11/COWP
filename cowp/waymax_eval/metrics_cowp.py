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
    """Compute label-space diagnostics, not simulator collision metrics.

    ``conventional_safe`` is a candidate-label predicate.  Its complement must
    not be presented as an executed closed-loop collision rate.  The legacy CR
    alias is retained for old table scripts but is paired with explicit protocol
    markers and a descriptive metric name.
    """
    n = len(selected_indices)
    cf_count = 0
    false_safe = 0
    cbs = []
    oprs = []
    hbcr = 0
    collision_or_offroad = 0
    fallback_count = 0
    certificate_eval_count = 0
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
        crit_selected = label["cowp/critical/valid"].astype(bool)
        mech = np.asarray(label.get("cowp/critical/mechanism_valid", crit_selected), dtype=bool)
        crit = crit_selected & mech
        cand_valid = np.asarray(label["cowp/candidates/valid"], dtype=bool)
        cert_valid = np.asarray(label.get("cowp/candidates/certificate_valid", cand_valid), dtype=bool) & cand_valid
        conv = bool(label["cowp/candidates/conventional_safe"][k])
        collision_or_offroad += int(not conv)
        traj = label["cowp/candidates/trajectory"][k]
        p_m = _trajectory_progress_m(traj)
        progress_m.append(p_m)
        # Paper EP is route/progress normalized.  In label-only evaluation we do
        # not have the route integral, so normalize by the best valid/conventional
        # lattice progress and keep EP_m for debugging.
        progress_norm.append(float(np.clip(p_m / ref_progress, 0.0, 1.0)))
        if k >= len(cert_valid) or not bool(cert_valid[k]):
            continue
        certificate_eval_count += 1
        wit = label["cowp/witness/exists"][k].astype(bool) & crit
        cf_count += int(conv)
        false_safe += int(conv and np.any(wit))
        if np.any(crit):
            num_candidates = int(np.asarray(label["cowp/candidates/trajectory"]).shape[0])
            burden = np.asarray(
                label.get(
                    "cowp/witness/burden_total",
                    np.zeros((num_candidates, len(crit)), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            opr = np.asarray(
                label.get(
                    "cowp/witness/opr",
                    np.ones((num_candidates, len(crit)), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            cbs.append(float(np.nanmax(np.nan_to_num(burden[k, crit], nan=0.0, posinf=0.0, neginf=0.0))))
            oprs.append(float(np.nanmean(np.nan_to_num(opr[k, crit], nan=0.0, posinf=1.0, neginf=0.0))))
            min_safe = np.asarray(label.get("cowp/witness/min_safe_burden", burden), dtype=np.float32)
            beta = np.asarray(label.get("cowp/natural/beta", np.full(len(crit), 0.65, dtype=np.float32)), dtype=np.float32)
            if beta.shape[0] < crit.shape[0]:
                beta = np.pad(beta, (0, crit.shape[0] - beta.shape[0]), constant_values=0.65)
            hbcr += int(np.any(min_safe[k, crit] > beta[: crit.shape[0]][crit]))
    conventional_unsafe_rate = float(collision_or_offroad / max(n, 1))
    return {
        "MetricProtocol/LabelSpaceOnly": 1.0,
        "MetricProtocol/ClosedLoopCollisionAvailable": 0.0,
        "OfflineConventionalUnsafeRate": conventional_unsafe_rate,
        "CR_proxy_deprecated": conventional_unsafe_rate,
        # Backward-compatible alias only.  Paper tables must use Waymax standard
        # metrics for CR and the descriptive key above for label-space analysis.
        "CR": conventional_unsafe_rate,
        "EP": float(np.mean(progress_norm)) if progress_norm else 0.0,
        "EP_m": float(np.mean(progress_m)) if progress_m else 0.0,
        "FallbackRate": float(fallback_count / max(n, 1)),
        "FSR": float(false_safe / max(cf_count, 1)),
        "CBS": float(np.mean(cbs)) if cbs else 0.0,
        "OPR": float(np.mean(oprs)) if oprs else 0.0,
        "HBCR": float(hbcr / max(certificate_eval_count, 1)),
        "CertificateLabelCoverage/SelectedRate": float(certificate_eval_count / max(n, 1)),
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
        cert_valid = np.asarray(label.get("cowp/candidates/certificate_valid", cand_valid), dtype=bool) & cand_valid
        ncf = label["cowp/candidates/noncoercive_feasible"].astype(bool) & cert_valid
        false_safe = label["cowp/candidates/false_safe"].astype(bool) & cert_valid
        accepted = accepted_mask.astype(bool) if accepted_mask is not None else np.zeros_like(cand_valid)
        noncoercive_total += int(ncf.sum())
        noncoercive_accepted += int((accepted & ncf).sum())
        false_safe_total += int(false_safe.sum())
        false_safe_accepted += int((accepted & false_safe).sum())
        gt = label["cowp/witness/exists"].astype(bool)
        pred = gt.copy()  # rule certificate mode uses deterministic labels as predictions.
        crit = label["cowp/critical/valid"].astype(bool)
        mech = np.asarray(label.get("cowp/critical/mechanism_valid", crit), dtype=bool) & crit
        mask = cand_valid[:, None] & mech[None, :]
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
        crit = np.asarray(label.get("cowp/critical/valid", np.ones(exists.shape[1], dtype=bool)), dtype=bool)
        mech = np.asarray(label.get("cowp/critical/mechanism_valid", crit), dtype=bool) & crit
        exists = exists & mech[None, :]
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
    max_w = np.nan_to_num(np.asarray([float(r.get("max_witness_prob", 0.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    threshold = np.nan_to_num(np.asarray([float(r.get("witness_threshold", 0.5)) for r in rows], dtype=np.float32), nan=0.5, posinf=0.5, neginf=0.5)
    min_opr = np.nan_to_num(np.asarray([float(r.get("min_opr", 1.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    mean_opr = np.nan_to_num(np.asarray([float(r.get("mean_opr", 1.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    burden = np.nan_to_num(np.asarray([float(r.get("max_predicted_burden", 0.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=2.0, neginf=0.0)
    fallback = np.asarray([bool(r.get("fallback_used", False)) for r in rows], dtype=bool)
    out = {
        # Explicit protocol marker: these values are generated by the same model
        # that selects the plan.  They are useful health diagnostics, but are not
        # counterfactual ground truth and must not be reported as causal evidence.
        "ClosedLoopMechanismProxyOnly": 1.0,
        "ClosedLoopMechanismGroundTruthAvailable": 0.0,
        "ClosedLoopPredFSR": float(np.mean(max_w >= threshold)),
        "ClosedLoopMeanWitnessProb": float(np.mean(max_w)),
        "ClosedLoopCBS_pred": float(np.mean(burden)),
        "ClosedLoopOPR_min": float(np.mean(min_opr)),
        "ClosedLoopOPR_mean": float(np.mean(mean_opr)),
        "ClosedLoopFallbackStepRate": float(np.mean(fallback)),
        "ClosedLoopPolicySteps": float(len(rows)),
    }
    # Extra online-health diagnostics.  They make it obvious whether failures are
    # caused by candidate starvation, witness over-rejection, missing critical
    # agents, or missing conflict/map tokens.
    for key in ("accepted_candidates", "valid_candidates", "conventional_candidates", "critical_agents", "conflict_tokens"):
        vals = np.asarray([float(r.get(key, 0.0)) for r in rows], dtype=np.float32)
        out[f"ClosedLoopMean/{key}"] = float(np.mean(vals)) if vals.size else 0.0
    for key in (
        "selected_candidate_ncf_prob",
        "selected_candidate_false_safe_prob",
        "selected_candidate_quality_prob",
        "selected_candidate_cert_risk",
        "selected_candidate_pressure_prior",
        "selected_candidate_rule_risk",
        "selected_candidate_action_risk",
        "selected_outcome_risk",
        "selected_outcome_decision_risk",
        "min_candidate_cert_risk",
        "mean_candidate_cert_risk",
        "mean_candidate_pressure_prior",
        "mean_candidate_rule_risk",
        "mean_candidate_action_risk",
        "selected_plan_continuity_risk",
    ):
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"ClosedLoopMean/{key}"] = float(np.mean(vals))
    for key in (
        "runtime_state_extract_map_s",
        "runtime_candidate_build_cpu_s",
        "runtime_h2d_s",
        "runtime_model_forward_s",
        "runtime_selection_s",
        "runtime_action_projection_s",
        "runtime_policy_total_s",
    ):
        vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"ClosedLoopPolicyRuntimeMean/{key}"] = float(np.mean(vals))
            out[f"ClosedLoopPolicyRuntimeSum/{key}"] = float(np.sum(vals))
    runtime_total = out.get("ClosedLoopPolicyRuntimeSum/runtime_policy_total_s")
    if runtime_total and float(runtime_total) > 0.0:
        for key in (
            "runtime_state_extract_map_s", "runtime_candidate_build_cpu_s", "runtime_h2d_s",
            "runtime_model_forward_s", "runtime_selection_s", "runtime_action_projection_s",
        ):
            val = out.get(f"ClosedLoopPolicyRuntimeSum/{key}")
            if val is not None:
                out[f"ClosedLoopPolicyRuntimeFraction/{key}"] = float(val) / float(runtime_total)
    reasons = [str(r.get("fallback_reason", "none")) for r in rows]
    for reason in sorted(set(reasons)):
        out[f"ClosedLoopFallbackReason/{reason}"] = float(sum(x == reason for x in reasons) / max(len(reasons), 1))
    return out


def policy_diagnostic_episode_summary(rollouts: list[dict]) -> dict[str, float]:
    """Episode-level COWP-style online diagnostic summary.

    The paper's FSR/CBS/OPR/HBCR are counterfactual burden metrics.  During the
    Waymax online path we do not have the full pseudo-label counterfactual bank
    for every executed state, so this summary uses the policy's predicted witness
    and burden heads as an online proxy.  Ground-truth COWP metrics remain the
    label/offline path; this function makes the closed-loop output interpretable
    at episode granularity.
    """
    if not rollouts:
        return {}
    episodes_with_rows = 0
    collision_free = 0
    pred_false_safe = 0
    cbs_vals: list[float] = []
    opr_min_vals: list[float] = []
    opr_mean_vals: list[float] = []
    hbcr = 0
    fallback_episode = 0
    for item in rollouts:
        rows = item.get("policy_diagnostics", []) or []
        if not rows:
            continue
        episodes_with_rows += 1
        row_max_w = np.nan_to_num(np.asarray([float(r.get("max_witness_prob", 0.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        row_thr = np.nan_to_num(np.asarray([float(r.get("witness_threshold", 0.5)) for r in rows], dtype=np.float32), nan=0.5, posinf=0.5, neginf=0.5)
        row_burden = np.nan_to_num(np.asarray([float(r.get("max_predicted_burden", 0.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=2.0, neginf=0.0)
        row_opr_min = np.nan_to_num(np.asarray([float(r.get("min_opr", 1.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        row_opr_mean = np.nan_to_num(np.asarray([float(r.get("mean_opr", 1.0)) for r in rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        max_w = float(row_max_w.max())
        threshold = float(row_thr.max())
        max_burden = float(row_burden.max())
        beta = max(float(r.get("beta_threshold", 0.65)) for r in rows)
        min_opr = float(row_opr_min.min())
        mean_opr = float(row_opr_mean.mean())
        fallback_episode += int(any(bool(r.get("fallback_used", False)) for r in rows))
        cbs_vals.append(max_burden)
        opr_min_vals.append(min_opr)
        opr_mean_vals.append(mean_opr)
        hbcr += int(max_burden > beta)
        std = item.get("standard_metrics", {}) or {}
        # If standard metrics are unavailable, treat the episode as eligible for
        # predicted FSR rather than silently dropping all episodes.
        is_collision_free = float(std.get("CR", 0.0)) <= 0.0
        collision_free += int(is_collision_free)
        pred_false_safe += int(is_collision_free and max_w >= threshold)
    if episodes_with_rows == 0:
        return {}
    return {
        "PredFSR_episode": float(pred_false_safe / max(collision_free, 1)),
        "PredCBS_episode": float(np.mean(cbs_vals)) if cbs_vals else 0.0,
        "PredOPR_min_episode": float(np.mean(opr_min_vals)) if opr_min_vals else 0.0,
        "PredOPR_mean_episode": float(np.mean(opr_mean_vals)) if opr_mean_vals else 0.0,
        "PredHBCR_episode": float(hbcr / max(episodes_with_rows, 1)),
        "FallbackEpisodeRate": float(fallback_episode / max(episodes_with_rows, 1)),
        "EpisodesWithDiagnostics": float(episodes_with_rows),
    }




def module_effect_metrics(
    label_dicts: list[dict[str, np.ndarray]],
    cfg: dict,
    methods: list[str] | None = None,
    *,
    precomputed_decisions: dict[str, list[tuple[int, np.ndarray]]] | None = None,
    precomputed_selected: dict[str, list[int]] | None = None,
    precomputed_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Quantify whether the paper modules change decisions in label space.

    This is intentionally label/certificate based.  It answers: if we disable a
    module, do accepted masks and selected candidates change, and do false-safe /
    option metrics degrade?  These numbers are useful before running expensive
    Waymax closed-loop experiments.
    """
    from cowp.waymax_eval.baselines import planner_for_method

    methods = methods or [
        "cowp",
        "cowp_wo_counterfactual",
        "cowp_wo_neutral_branch",
        "cowp_wo_priority_branch",
        "cowp_wo_option_preservation",
        "cowp_wo_witness_rejection",
        "soft_burden_cost_only",
    ]
    decisions: dict[str, list[tuple[int, np.ndarray]]] = dict(precomputed_decisions or {})
    selected: dict[str, list[int]] = dict(precomputed_selected or {})
    metrics: dict[str, dict[str, float]] = {k: dict(v) for k, v in (precomputed_metrics or {}).items()}
    for method in methods:
        if method in decisions and method in selected and method in metrics:
            continue
        planner = planner_for_method(method, cfg)
        rows: list[tuple[int, np.ndarray]] = []
        idxs: list[int] = []
        for label in label_dicts:
            dec = planner.select_from_labels(label)
            rows.append((dec.candidate_index, dec.accepted_mask.astype(bool)))
            idxs.append(int(dec.candidate_index))
        decisions[method] = rows
        selected[method] = idxs
        metrics[method] = metrics_from_labels(idxs, label_dicts)
    full = decisions.get("cowp")
    if full is not None:
        for method, rows in decisions.items():
            if method == "cowp":
                metrics[method]["DecisionChangeVsFull"] = 0.0
                metrics[method]["AcceptedJaccardVsFull"] = 1.0
                continue
            changes = 0
            jaccards = []
            for (k_full, mask_full), (k_m, mask_m) in zip(full, rows):
                changes += int(k_full != k_m)
                union = np.logical_or(mask_full, mask_m).sum()
                inter = np.logical_and(mask_full, mask_m).sum()
                jaccards.append(float(inter / max(union, 1)))
            metrics[method]["DecisionChangeVsFull"] = float(changes / max(len(rows), 1))
            metrics[method]["AcceptedJaccardVsFull"] = float(np.mean(jaccards)) if jaccards else 1.0
    return metrics


def policy_diagnostic_scenario_rows(rollouts: list[dict]) -> list[dict]:
    """Compact per-episode rows for attributing physical failures to fallback/selection.

    These rows intentionally contain only aggregated online diagnostics plus Waymax
    event metrics; they are much smaller than serializing every policy step.
    """
    out: list[dict] = []
    for idx, item in enumerate(rollouts):
        rows = item.get("policy_diagnostics", []) or []
        std = item.get("standard_metrics", {}) or {}
        n = len(rows)
        fallback = np.asarray([bool(r.get("fallback_used", False)) for r in rows], dtype=bool) if n else np.asarray([], dtype=bool)
        emergency = np.asarray([bool(r.get("emergency_action_used", False)) for r in rows], dtype=bool) if n else np.asarray([], dtype=bool)
        reasons = [str(r.get("fallback_reason", "none")) for r in rows]
        valid_counts = np.asarray([int(r.get("valid_candidates", 0)) for r in rows], dtype=np.int64) if n else np.asarray([], dtype=np.int64)
        conventional_counts = np.asarray([int(r.get("conventional_candidates", 0)) for r in rows], dtype=np.int64) if n else np.asarray([], dtype=np.int64)
        roadgraph_counts = np.asarray([int(r.get("roadgraph_safe_candidates", 0)) for r in rows], dtype=np.int64) if n else np.asarray([], dtype=np.int64)
        collision_counts = np.asarray([int(r.get("collision_safe_candidates", 0)) for r in rows], dtype=np.int64) if n else np.asarray([], dtype=np.int64)
        zero_conv_reasons = [str(r.get("zero_conventional_reason", "none")) for r in rows]

        def _mean(key: str, *, mask: np.ndarray | None = None) -> float | None:
            if not rows:
                return None
            vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
            valid = np.isfinite(vals)
            if mask is not None:
                valid &= mask
            return float(vals[valid].mean()) if valid.any() else None

        def _mean_abs(key: str, *, mask: np.ndarray | None = None) -> float | None:
            if not rows:
                return None
            vals = np.asarray([abs(float(r.get(key, np.nan))) for r in rows], dtype=np.float64)
            valid = np.isfinite(vals)
            if mask is not None:
                valid &= mask
            return float(vals[valid].mean()) if valid.any() else None

        first_fallback = next((j for j, r in enumerate(rows) if bool(r.get("fallback_used", False))), None)
        first_emergency = next((j for j, r in enumerate(rows) if bool(r.get("emergency_action_used", False))), None)
        first_zero_valid = next((j for j, count in enumerate(valid_counts) if int(count) <= 0), None)
        first_zero_conventional = next((j for j, count in enumerate(conventional_counts) if int(count) <= 0), None)
        rec: dict[str, object] = {
            "scenario_id": str(item.get("scenario_id", idx)),
            "steps": int(item.get("steps", n)),
            "fallback_step_rate": float(fallback.mean()) if fallback.size else 0.0,
            "fallback_episode": bool(fallback.any()) if fallback.size else False,
            "first_fallback_policy_step": int(first_fallback) if first_fallback is not None else None,
            "emergency_action_step_rate": float(emergency.mean()) if emergency.size else 0.0,
            "emergency_action_episode": bool(emergency.any()) if emergency.size else False,
            "first_emergency_policy_step": int(first_emergency) if first_emergency is not None else None,
            "zero_valid_candidate_step_rate": float((valid_counts <= 0).mean()) if valid_counts.size else 0.0,
            "zero_conventional_candidate_step_rate": float((conventional_counts <= 0).mean()) if conventional_counts.size else 0.0,
            "first_zero_valid_candidate_policy_step": int(first_zero_valid) if first_zero_valid is not None else None,
            "first_zero_conventional_candidate_policy_step": int(first_zero_conventional) if first_zero_conventional is not None else None,
            "no_certificate_step_rate": float(sum(x == "no_certificate_use_least_coercive_conventional" for x in reasons) / max(n, 1)),
            "no_conventional_step_rate": float(sum(x in {"no_conventional_use_least_coercive_valid", "no_conventional_use_recursive_viability", "no_conventional_use_rvr_pareto_guard", "no_conventional_use_successor_option_viability", "no_conventional_use_bihorizon_option_viability", "no_conventional_use_successor_restore_only", "no_conventional_use_trihorizon_option_persistence", "no_conventional_use_sov_recovery_commitment", "no_conventional_use_sov_dominance_hysteresis", "no_conventional_use_recovery_option_spectrum_hysteresis", "no_conventional_use_transition_guarded_rosh", "no_conventional_use_executable_option_spectrum_hysteresis", "no_conventional_use_waymax_kinematic_guarded_rosh", "no_conventional_use_control_projected_option_spectrum_hysteresis", "no_conventional_use_control_projected_recovery_frontier", "no_conventional_use_recourse_returnability_bridge"} for x in reasons) / max(n, 1)),
            "recursive_viability_recovery_step_rate": float(sum(x == "no_conventional_use_recursive_viability" for x in reasons) / max(n, 1)),
            "rvr_pareto_guard_step_rate": float(sum(x == "no_conventional_use_rvr_pareto_guard" for x in reasons) / max(n, 1)),
            "successor_option_viability_step_rate": float(sum(x == "no_conventional_use_successor_option_viability" for x in reasons) / max(n, 1)),
            "bihorizon_option_viability_step_rate": float(sum(x == "no_conventional_use_bihorizon_option_viability" for x in reasons) / max(n, 1)),
            "successor_restore_only_step_rate": float(sum(x == "no_conventional_use_successor_restore_only" for x in reasons) / max(n, 1)),
            "trihorizon_option_persistence_step_rate": float(sum(x == "no_conventional_use_trihorizon_option_persistence" for x in reasons) / max(n, 1)),
            "sov_recovery_commitment_step_rate": float(sum(x == "no_conventional_use_sov_recovery_commitment" for x in reasons) / max(n, 1)),
            "sov_dominance_hysteresis_step_rate": float(sum(x == "no_conventional_use_sov_dominance_hysteresis" for x in reasons) / max(n, 1)),
            "recovery_option_spectrum_hysteresis_step_rate": float(sum(x == "no_conventional_use_recovery_option_spectrum_hysteresis" for x in reasons) / max(n, 1)),
            "transition_guarded_rosh_step_rate": float(sum(x == "no_conventional_use_transition_guarded_rosh" for x in reasons) / max(n, 1)),
            "executable_option_spectrum_hysteresis_step_rate": float(sum(x == "no_conventional_use_executable_option_spectrum_hysteresis" for x in reasons) / max(n, 1)),
            "waymax_kinematic_guarded_rosh_step_rate": float(sum(x == "no_conventional_use_waymax_kinematic_guarded_rosh" for x in reasons) / max(n, 1)),
            "control_projected_option_spectrum_hysteresis_step_rate": float(sum(x == "no_conventional_use_control_projected_option_spectrum_hysteresis" for x in reasons) / max(n, 1)),
            "control_projected_recovery_frontier_step_rate": float(sum(x == "no_conventional_use_control_projected_recovery_frontier" for x in reasons) / max(n, 1)),
            "recourse_returnability_bridge_step_rate": float(sum(x == "no_conventional_use_recourse_returnability_bridge" for x in reasons) / max(n, 1)),
            "recovery_switch_step_rate": _mean("recovery_switch_applied"),
            "successor_option_probe_step_rate": _mean("successor_option_probe_used"),
            "second_successor_option_probe_step_rate": _mean("second_successor_option_probe_used"),
            "mean_successor_signature_compare_on_probes": _mean("successor_signature_compare", mask=np.asarray([bool(r.get("successor_option_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_second_successor_signature_compare_on_probes": _mean("second_successor_signature_compare", mask=np.asarray([bool(r.get("second_successor_option_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_commitment_active_step_rate": _mean("recovery_commitment_active_after"),
            "recovery_commitment_entry_step_rate": _mean("recovery_commitment_entered"),
            "recovery_commitment_continue_step_rate": _mean("recovery_commitment_continued"),
            "recovery_commitment_clear_step_rate": _mean("recovery_commitment_cleared"),
            "recovery_hysteresis_active_step_rate": _mean("recovery_hysteresis_active_after"),
            "recovery_hysteresis_entry_step_rate": _mean("recovery_hysteresis_entered"),
            "recovery_hysteresis_continue_step_rate": _mean("recovery_hysteresis_continued"),
            "recovery_hysteresis_exit_step_rate": _mean("recovery_hysteresis_exited"),
            "recovery_hysteresis_clear_step_rate": _mean("recovery_hysteresis_cleared"),
            "recovery_option_profile_probe_step_rate": _mean("recovery_option_profile_probe_used"),
            "recovery_option_profile_strict_dominance_rate_on_probes": _mean("recovery_option_profile_strict_dominates", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_option_profile_weak_dominance_rate_on_probes": _mean("recovery_option_profile_weak_dominates", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_option_profile_min_margin_on_probes": _mean("recovery_option_profile_min_margin", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_option_profile_area_delta_on_probes": _mean("recovery_option_profile_area_delta", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_executable_option_profile_probe_step_rate": _mean("recovery_executable_option_profile_used"),
            "recovery_base_controller_transition_feasible_rate_on_probes": _mean("recovery_base_controller_transition_feasible", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_rvr_controller_transition_feasible_rate_on_probes": _mean("recovery_rvr_controller_transition_feasible", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_controller_transition_delta_on_probes": _mean("recovery_controller_transition_delta", mask=np.asarray([bool(r.get("recovery_option_profile_probe_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_base_transition_feasible_candidates_on_exec_probes": _mean("recovery_option_profile_base_transition_feasible_candidates", mask=np.asarray([bool(r.get("recovery_executable_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_rvr_transition_feasible_candidates_on_exec_probes": _mean("recovery_option_profile_rvr_transition_feasible_candidates", mask=np.asarray([bool(r.get("recovery_executable_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_base_transition_rejected_roadgraph_candidates_on_exec_probes": _mean("recovery_option_profile_base_transition_rejected_roadgraph_candidates", mask=np.asarray([bool(r.get("recovery_executable_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_rvr_transition_rejected_roadgraph_candidates_on_exec_probes": _mean("recovery_option_profile_rvr_transition_rejected_roadgraph_candidates", mask=np.asarray([bool(r.get("recovery_executable_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "selected_controller_transition_feasible_step_rate": _mean("selected_controller_transition_feasible"),
            "recovery_waymax_kinematic_guard_probe_step_rate": _mean("recovery_waymax_kinematic_guard_used"),
            "recovery_control_projected_option_profile_probe_step_rate": _mean("recovery_control_projected_option_profile_used"),
            "recovery_base_waymax_kinematic_feasible_rate_on_probes": _mean("recovery_base_waymax_kinematic_feasible", mask=np.asarray([bool(r.get("recovery_waymax_kinematic_guard_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_rvr_waymax_kinematic_feasible_rate_on_probes": _mean("recovery_rvr_waymax_kinematic_feasible", mask=np.asarray([bool(r.get("recovery_waymax_kinematic_guard_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_waymax_kinematic_transition_delta_on_probes": _mean("recovery_waymax_kinematic_transition_delta", mask=np.asarray([bool(r.get("recovery_waymax_kinematic_guard_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_base_waymax_abs_steering_curvature_on_probes": _mean_abs("recovery_base_waymax_steering_curvature", mask=np.asarray([bool(r.get("recovery_waymax_kinematic_guard_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_rvr_waymax_abs_steering_curvature_on_probes": _mean_abs("recovery_rvr_waymax_steering_curvature", mask=np.asarray([bool(r.get("recovery_waymax_kinematic_guard_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_base_control_projected_h1_kinematic_feasible_candidates": _mean("recovery_option_profile_base_control_projected_h1_kinematic_feasible_candidates", mask=np.asarray([bool(r.get("recovery_control_projected_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_rvr_control_projected_h1_kinematic_feasible_candidates": _mean("recovery_option_profile_rvr_control_projected_h1_kinematic_feasible_candidates", mask=np.asarray([bool(r.get("recovery_control_projected_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_base_control_projected_full_kinematic_feasible_candidates": _mean("recovery_option_profile_base_control_projected_full_kinematic_feasible_candidates", mask=np.asarray([bool(r.get("recovery_control_projected_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "mean_recovery_rvr_control_projected_full_kinematic_feasible_candidates": _mean("recovery_option_profile_rvr_control_projected_full_kinematic_feasible_candidates", mask=np.asarray([bool(r.get("recovery_control_projected_option_profile_used", False)) for r in rows], dtype=bool)) if n else None,
            "recovery_frontier_probe_step_rate": _mean("recovery_frontier_probe_used"),
            "mean_recovery_frontier_representative_count_on_probes": _mean(
                "recovery_frontier_representative_count",
                mask=np.asarray([bool(r.get("recovery_frontier_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_profiles_evaluated_on_probes": _mean(
                "recovery_frontier_profiles_evaluated",
                mask=np.asarray([bool(r.get("recovery_frontier_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_strict_admissible_count_on_probes": _mean(
                "recovery_frontier_strict_admissible_count",
                mask=np.asarray([bool(r.get("recovery_frontier_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_weak_admissible_count_on_probes": _mean(
                "recovery_frontier_weak_admissible_count",
                mask=np.asarray([bool(r.get("recovery_frontier_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_current_prefix_admissible_count_on_probes": _mean(
                "recovery_frontier_current_prefix_admissible_count",
                mask=np.asarray([bool(r.get("recovery_frontier_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_selected_prefix_delta_on_switches": _mean(
                "recovery_frontier_selected_prefix_delta_steps",
                mask=np.asarray([bool(r.get("recovery_switch_applied", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recovery_frontier_selected_non_rvr_rate_on_switches": _mean(
                "recovery_frontier_selected_is_non_rvr",
                mask=np.asarray([bool(r.get("recovery_switch_applied", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recovery_frontier_selected_historical_rvr_rate_on_switches": _mean(
                "recovery_frontier_selected_is_historical_rvr",
                mask=np.asarray([bool(r.get("recovery_switch_applied", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_frontier_selected_fallback_score_delta_on_switches": _mean(
                "recovery_frontier_selected_fallback_score_delta",
                mask=np.asarray([bool(r.get("recovery_switch_applied", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recourse_returnability_probe_step_rate": _mean("recourse_returnability_probe_used"),
            "recourse_returnability_strict_dominance_rate_on_probes": _mean(
                "recourse_returnability_strict_dominates",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recourse_base_direct_restore_rate_on_probes": _mean(
                "recourse_base_direct_restore",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recourse_rvr_direct_restore_rate_on_probes": _mean(
                "recourse_rvr_direct_restore",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_base_macro_count_on_probes": _mean(
                "recourse_base_macro_count",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_rvr_macro_count_on_probes": _mean(
                "recourse_rvr_macro_count",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recourse_current_action_survival_step_rate": _mean("recourse_current_action_survives_one_step"),
            "mean_recourse_base_action_classes_available_on_probes": _mean(
                "recourse_base_action_classes_available",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_rvr_action_classes_available_on_probes": _mean(
                "recourse_rvr_action_classes_available",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_base_action_classes_evaluated_on_probes": _mean(
                "recourse_base_action_classes_evaluated",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_rvr_action_classes_evaluated_on_probes": _mean(
                "recourse_rvr_action_classes_evaluated",
                mask=np.asarray([bool(r.get("recourse_returnability_probe_used", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recovery_bridge_pending_step_rate": _mean("recovery_bridge_pending_after"),
            "recovery_bridge_entry_step_rate": _mean("recovery_bridge_entered"),
            "recovery_bridge_direct_entry_step_rate": _mean("recovery_bridge_direct_entry"),
            "recovery_bridge_recourse_execution_step_rate": _mean("recovery_bridge_recourse_executed"),
            "recovery_bridge_abort_step_rate": _mean("recovery_bridge_aborted"),
            "mean_direct_restoring_candidate_count_on_bridge_steps": _mean(
                "recourse_direct_restoring_candidate_count",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_bridge_allowed_macro_count_on_bridge_steps": _mean(
                "recovery_bridge_allowed_macro_count_before",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_bridge_candidate_pool_on_bridge_steps": _mean(
                "recourse_bridge_candidate_pool",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_bridge_action_classes_available_on_bridge_steps": _mean(
                "recourse_bridge_action_classes_available",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_bridge_action_classes_evaluated_on_bridge_steps": _mean(
                "recourse_bridge_representatives_evaluated",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recourse_bridge_minimum_prefix_steps_on_bridge_steps": _mean(
                "recourse_bridge_minimum_prefix_steps",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recovery_bridge_execution_rate_on_bridge_steps": _mean(
                "recovery_bridge_recourse_executed",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "recovery_bridge_abort_rate_on_bridge_steps": _mean(
                "recovery_bridge_aborted",
                mask=np.asarray([bool(r.get("recovery_bridge_pending_before", False)) for r in rows], dtype=bool),
            ) if n else None,
            "selected_waymax_kinematic_feasible_step_rate": _mean("selected_waymax_kinematic_feasible"),
            "selected_waymax_kinematic_feasible_rate_on_recovery_switch_steps": _mean(
                "selected_waymax_kinematic_feasible",
                mask=np.asarray([bool(r.get("recovery_switch_applied", False)) for r in rows], dtype=bool),
            ) if n else None,
            "mean_recovery_prefix_gain_steps": _mean("recovery_prefix_gain_steps"),
            "mean_recovery_action_risk_delta": _mean("recovery_action_risk_delta"),
            "mean_recovery_rule_risk_delta": _mean("recovery_rule_risk_delta"),
            "mean_recovery_pressure_risk_delta": _mean("recovery_pressure_risk_delta"),
            "zero_conventional_collision_empty_step_rate": float(sum(x == "collision_empty" for x in zero_conv_reasons) / max(n, 1)),
            "zero_conventional_roadgraph_empty_step_rate": float(sum(x == "roadgraph_empty" for x in zero_conv_reasons) / max(n, 1)),
            "zero_conventional_both_empty_step_rate": float(sum(x == "road_and_collision_empty" for x in zero_conv_reasons) / max(n, 1)),
            "zero_conventional_intersection_empty_step_rate": float(sum(x == "intersection_empty" for x in zero_conv_reasons) / max(n, 1)),
            "no_valid_step_rate": float(sum(x in {"no_valid_candidate", "baseline_no_valid_emergency_stop"} for x in reasons) / max(n, 1)),
            "accepted_priority_ncf_step_rate": float(sum(x == "accepted_priority_ncf" for x in reasons) / max(n, 1)),
            "mean_accepted_candidates": _mean("accepted_candidates"),
            "mean_conventional_candidates": _mean("conventional_candidates"),
            "mean_roadgraph_safe_candidates": _mean("roadgraph_safe_candidates"),
            "mean_collision_safe_candidates": _mean("collision_safe_candidates"),
            "mean_max_collision_safe_prefix_steps": _mean("max_collision_safe_prefix_steps"),
            "mean_selected_collision_safe_prefix_steps": _mean("selected_collision_safe_prefix_steps"),
            "mean_valid_candidates": _mean("valid_candidates"),
            "mean_selected_action_risk": _mean("selected_candidate_action_risk"),
            "mean_selected_rule_risk": _mean("selected_candidate_rule_risk"),
            "mean_selected_cert_risk": _mean("selected_candidate_cert_risk"),
            "mean_selected_outcome_risk": _mean("selected_outcome_risk"),
            "fallback_mean_selected_outcome_risk": _mean("selected_outcome_risk", mask=fallback) if fallback.size else None,
            "accepted_mean_selected_outcome_risk": _mean("selected_outcome_risk", mask=~fallback) if fallback.size else None,
        }
        for key in (
            "CR", "CollisionRate", "OffroadRate", "KinematicsInfeasibilityRate", "EP",
            "WaymaxAny/OverlapMetric", "WaymaxAny/OffroadMetric", "WaymaxAny/KinematicsInfeasibilityMetric",
            "FirstPositiveStep/OverlapMetric", "FirstPositiveStep/OffroadMetric", "FirstPositiveStep/KinematicsInfeasibilityMetric",
        ):
            if key in std:
                rec[key] = std[key]

        # How much of the policy history before the first physical event was already
        # fallback?  This is more informative than episode-level co-occurrence when
        # fallback happens in >90% of episodes.
        event_map = {
            "collision": "FirstPositiveStep/OverlapMetric",
            "offroad": "FirstPositiveStep/OffroadMetric",
            "kinematics": "FirstPositiveStep/KinematicsInfeasibilityMetric",
        }
        for name, first_key in event_map.items():
            if first_key in std and fallback.size:
                first_metric_step = max(int(std[first_key]), 1)  # accumulator is 1-indexed after env step
                prefix = fallback[: min(first_metric_step, fallback.size)]
                action_idx = min(first_metric_step - 1, fallback.size - 1)
                action_row = rows[action_idx] if action_idx < len(rows) else {}
                rec[f"fallback_rate_before_first_{name}"] = float(prefix.mean()) if prefix.size else 0.0
                rec[f"fallback_at_action_before_first_{name}"] = bool(fallback[action_idx])
                rec[f"emergency_action_at_action_before_first_{name}"] = bool(action_row.get("emergency_action_used", False))
                rec[f"execution_trajectory_source_at_action_before_first_{name}"] = str(action_row.get("execution_trajectory_source", "candidate"))
                rec[f"selected_candidate_valid_at_action_before_first_{name}"] = bool(action_row.get("selected_candidate_valid", False))
                rec[f"selected_conventional_safe_at_action_before_first_{name}"] = bool(action_row.get("selected_candidate_conventional_safe", False))
                rec[f"selected_roadgraph_safe_at_action_before_first_{name}"] = bool(action_row.get("selected_candidate_roadgraph_safe", False))
                rec[f"selected_collision_safe_at_action_before_first_{name}"] = bool(action_row.get("selected_candidate_collision_safe", False))
                rec[f"selected_collision_safe_prefix_steps_at_action_before_first_{name}"] = int(action_row.get("selected_collision_safe_prefix_steps", 0))
                rec[f"selected_collision_min_clearance_margin_m_at_action_before_first_{name}"] = float(action_row.get("selected_collision_min_clearance_margin_m", 0.0))
                rec[f"zero_conventional_reason_at_action_before_first_{name}"] = str(action_row.get("zero_conventional_reason", "none"))
                rec[f"selected_macro_type_at_action_before_first_{name}"] = int(action_row.get("selected_macro_type", -1))
                rec[f"selected_macro_name_at_action_before_first_{name}"] = str(action_row.get("selected_macro_name", "unknown"))
                rec[f"fallback_reason_at_action_before_first_{name}"] = str(action_row.get("fallback_reason", "none"))
        out.append(rec)
    return out


def physical_failure_attribution_summary(rollouts: list[dict]) -> dict[str, float]:
    """Episode-level association diagnostics between physical failures and fallback.

    This is attribution *localization*, not a causal claim.  It tells the next
    experiment whether to focus on accepted-plan selection or uncertified fallback.
    """
    rows = policy_diagnostic_scenario_rows(rollouts)
    if not rows:
        return {}
    out: dict[str, float] = {"Episodes": float(len(rows))}
    fb = np.asarray([float(r.get("fallback_step_rate", 0.0)) for r in rows], dtype=np.float64)
    no_cert = np.asarray([float(r.get("no_certificate_step_rate", 0.0)) for r in rows], dtype=np.float64)
    act = np.asarray([float(r.get("mean_selected_action_risk") or 0.0) for r in rows], dtype=np.float64)
    cert = np.asarray([float(r.get("mean_selected_cert_risk") or 0.0) for r in rows], dtype=np.float64)
    out["MeanFallbackStepRate"] = float(fb.mean())
    out["EpisodesFallbackMajorityRate"] = float((fb >= 0.5).mean())
    emergency = np.asarray([float(r.get("emergency_action_step_rate", 0.0)) for r in rows], dtype=np.float64)
    zero_valid = np.asarray([float(r.get("zero_valid_candidate_step_rate", 0.0)) for r in rows], dtype=np.float64)
    zero_conv = np.asarray([float(r.get("zero_conventional_candidate_step_rate", 0.0)) for r in rows], dtype=np.float64)
    out["MeanEmergencyActionStepRate"] = float(emergency.mean())
    out["MeanZeroValidCandidateStepRate"] = float(zero_valid.mean())
    out["MeanZeroConventionalCandidateStepRate"] = float(zero_conv.mean())

    event_keys = {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }
    for label, key in event_keys.items():
        event = np.asarray([float(r.get(key, 0.0)) > 0.0 for r in rows], dtype=bool)
        out[f"{label}/Rate"] = float(event.mean())
        if event.any():
            out[f"{label}/MeanFallbackStepRate_Pos"] = float(fb[event].mean())
            out[f"{label}/MeanNoCertificateStepRate_Pos"] = float(no_cert[event].mean())
            out[f"{label}/MeanActionRisk_Pos"] = float(act[event].mean())
            out[f"{label}/MeanCertRisk_Pos"] = float(cert[event].mean())
            before_key = {
                "Collision": "fallback_rate_before_first_collision",
                "Offroad": "fallback_rate_before_first_offroad",
                "Kinematics": "fallback_rate_before_first_kinematics",
            }.get(label)
            if before_key:
                vals = np.asarray([float(r.get(before_key, np.nan)) for r in rows], dtype=np.float64)
                vals = vals[event & np.isfinite(vals)]
                if vals.size:
                    out[f"{label}/MeanFallbackRateBeforeFirstEvent"] = float(vals.mean())
                suffix = label.lower()
                emergency_before = np.asarray([bool(r.get(f"emergency_action_at_action_before_first_{suffix}", False)) for r in rows], dtype=bool)
                out[f"{label}/EmergencyActionImmediatelyBeforeFirstEventRate"] = float(emergency_before[event].mean())
        if (~event).any():
            out[f"{label}/MeanFallbackStepRate_Neg"] = float(fb[~event].mean())
            out[f"{label}/MeanNoCertificateStepRate_Neg"] = float(no_cert[~event].mean())
            out[f"{label}/MeanActionRisk_Neg"] = float(act[~event].mean())
            out[f"{label}/MeanCertRisk_Neg"] = float(cert[~event].mean())
        high = fb >= 0.5
        if high.any():
            out[f"{label}/Rate_FallbackMajority"] = float(event[high].mean())
        if (~high).any():
            out[f"{label}/Rate_FallbackMinority"] = float(event[~high].mean())
    return out
