from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocrap.planning.selector import SelectionResult, calibrated_constrained_select, constrained_lcb_select, crisp_select


BASELINES = [
    "nominal",
    "log_replay",
    "idm_proxy",
    "mpc_proxy",
    "risk_aware",
    "backup_filter",
    "contingency",
    "oracle_filter",
    "ocrap",
    "ocrap_teacher",
]


@dataclass
class BaselineSelection:
    selected_index: int
    reason: str
    admitted: np.ndarray
    score: np.ndarray


def _best_by_score(score: np.ndarray, feasible: np.ndarray) -> int:
    feasible = np.asarray(feasible, dtype=bool)
    score = np.asarray(score, dtype=float)
    if feasible.any():
        idxs = np.where(feasible)[0]
        return int(idxs[np.argmax(score[idxs])])
    return int(np.argmax(score)) if score.size else 0


def _admit_then_utility(admitted: np.ndarray, utility: np.ndarray, nominal_index: int = 0) -> int:
    admitted = np.asarray(admitted, dtype=bool)
    utility = np.asarray(utility, dtype=float)
    if 0 <= nominal_index < len(utility) and admitted[nominal_index]:
        return int(nominal_index)
    if admitted.any():
        idxs = np.where(admitted)[0]
        return int(idxs[np.argmax(utility[idxs])])
    return _best_by_score(utility, np.ones_like(admitted, dtype=bool))


def _finite_or_proxy(primary: np.ndarray, proxy: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=float)
    out = primary.copy()
    bad = ~np.isfinite(out)
    if proxy is not None:
        p = np.asarray(proxy, dtype=float)
        if p.shape != out.shape:
            p = np.resize(p, out.shape)
        out[bad] = p[bad]
    bad = ~np.isfinite(out)
    if bad.any():
        fb = np.asarray(fallback, dtype=float)
        if fb.shape != out.shape:
            fb = np.resize(fb, out.shape)
        out[bad] = fb[bad]
    return out


def _strip_version_suffix(name: str) -> str:
    base, sep, version = str(name).rpartition("_v")
    return base if sep and version.isdigit() and base else str(name)


def _bucket_aliases(name: str | None) -> list[str]:
    if not name:
        return []
    raw = str(name)
    aliases = [raw]
    for p in ("test_", "val_", "train_"):
        if raw.startswith(p):
            aliases.append(raw[len(p):])
    # Treat normal shard versions such as test_safe_v2 as safe unless a more
    # specific override is provided.  This keeps regime-conditioned selector
    # knobs usable after rebuilding safe_v2/safe_v3.
    aliases.extend([_strip_version_suffix(x) for x in list(aliases)])
    aliases.extend([raw.replace("-", "_"), raw.replace("_", "-")])
    out: list[str] = []
    for x in aliases:
        if x and x not in out:
            out.append(x)
    return out



def _cfg_value(scfg: dict, key: str, default, bucket_name: str | None = None):
    """Read raw selection config with optional bucket/regime overrides."""
    value = scfg.get(key, default)
    for map_key in (f"{key}_by_bucket", f"{key}_by_regime"):
        mapping = scfg.get(map_key, None)
        if isinstance(mapping, dict):
            for alias in _bucket_aliases(bucket_name):
                if alias in mapping and mapping[alias] not in {None, ""}:
                    value = mapping[alias]
                    break
    return value

def _cfg_float(scfg: dict, key: str, default: float, bucket_name: str | None = None) -> float:
    """Read scalar selection config with optional bucket/regime overrides."""
    value = scfg.get(key, default)
    for map_key in (f"{key}_by_bucket", f"{key}_by_regime"):
        mapping = scfg.get(map_key, None)
        if isinstance(mapping, dict):
            for alias in _bucket_aliases(bucket_name):
                if alias in mapping and mapping[alias] not in {None, ""}:
                    value = mapping[alias]
                    break
    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_bool(scfg: dict, key: str, default: bool, bucket_name: str | None = None) -> bool:
    """Read boolean config with optional bucket/regime overrides."""
    value = scfg.get(key, default)
    for map_key in (f"{key}_by_bucket", f"{key}_by_regime"):
        mapping = scfg.get(map_key, None)
        if isinstance(mapping, dict):
            for alias in _bucket_aliases(bucket_name):
                if alias in mapping and mapping[alias] not in {None, ""}:
                    value = mapping[alias]
                    break
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_has_value(scfg: dict, key: str, bucket_name: str | None = None) -> bool:
    """Return True when a scalar or bucket-specific override is configured.

    This matters for optional parameters whose scalar default is None.  The
    previous code checked only ``selection.intervention_budget_rate`` before
    reading the bucket map, so ``intervention_budget_rate_by_bucket.safe=...``
    was silently ignored unless the global scalar was also set.
    """
    if scfg.get(key, None) not in {None, ""}:
        return True
    for map_key in (f"{key}_by_bucket", f"{key}_by_regime"):
        mapping = scfg.get(map_key, None)
        if isinstance(mapping, dict):
            for alias in _bucket_aliases(bucket_name):
                if alias in mapping and mapping[alias] not in {None, ""}:
                    return True
    return False


def select_baseline(
    method: str,
    utility: np.ndarray,
    pred_r_dep: np.ndarray,
    teacher_r_dep: np.ndarray,
    teacher_r_orc: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float,
    gamma_H: float,
    gamma_D: float,
    cfg: dict | None = None,
    *,
    pred_r_orc: np.ndarray | None = None,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    pred_drs: np.ndarray | None = None,
    pred_direct_value: np.ndarray | None = None,
    pred_direct_rank: np.ndarray | None = None,
    pred_direct_std: np.ndarray | None = None,
    pred_direct_opportunity: np.ndarray | None = None,
    pred_direct_harm: np.ndarray | None = None,
    candidate_macro_names=None,
) -> BaselineSelection:
    """Select one candidate with a paper-baseline-style rule.

    These baselines intentionally operate on existing dataset labels/features so
    they can be evaluated before adding a separate neural model for each
    baseline.  ``ocrap`` is the only method that uses predicted deployable
    recoverability; ``ocrap_teacher`` is an upper-bound diagnostic that uses the
    teacher deployable label directly.
    """
    cfg = cfg or {}
    method = str(method).lower()
    utility = np.asarray(utility, dtype=float)
    pred_r_dep = np.asarray(pred_r_dep, dtype=float)
    teacher_r_dep = np.asarray(teacher_r_dep, dtype=float)
    teacher_r_orc = np.asarray(teacher_r_orc, dtype=float)
    hard = np.asarray(hard, dtype=float)
    harm = np.asarray(harm, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    safe_mask = feasible & (hard <= gamma_H) & (harm <= gamma_D)
    oracle_signal = _finite_or_proxy(teacher_r_orc, pred_r_orc, pred_r_dep)

    if method == "ocrap":
        scfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
        selector_name = str(scfg.get("ocrap_selector", scfg.get("selector", "lcb_constrained"))).lower()
        bucket_name = str(scfg.get("active_bucket_name", scfg.get("regime_name", "")) or "")
        if selector_name in {"crisp", "hard", "hard_threshold"}:
            sel: SelectionResult = crisp_select(utility, pred_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
            return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, pred_r_dep)
        beta = _cfg_float(scfg, "lcb_beta", 0.10, bucket_name)
        if selector_name in {"calibrated", "calibrated_constrained", "soft_constrained", "budgeted_calibrated"}:
            sel = calibrated_constrained_select(
                utility, pred_r_dep, hard, harm, feasible,
                gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D,
                pred_gap=pred_gap, nominal_deviation=nominal_deviation,
                lcb_beta=beta,
                shortfall_penalty=_cfg_float(scfg, "calibrated_shortfall_penalty", 1.0, bucket_name),
                gap_penalty=_cfg_float(scfg, "calibrated_gap_penalty", 0.05, bucket_name),
                intervention_penalty=_cfg_float(scfg, "intervention_penalty", 0.03, bucket_name),
                deviation_penalty=_cfg_float(scfg, "deviation_penalty", 0.15, bucket_name),
                recovery_bonus=_cfg_float(scfg, "recovery_bonus", 0.02, bucket_name),
                admission_bonus=_cfg_float(scfg, "calibrated_admission_bonus", 0.02, bucket_name),
                nominal_slack=_cfg_float(scfg, "nominal_slack", 0.03, bucket_name),
                nominal_slack_gap_limit=_cfg_float(scfg, "nominal_slack_gap_limit", 0.50, bucket_name),
                nominal_utility_slack=_cfg_float(scfg, "nominal_utility_slack", 0.05, bucket_name),
                safe_nominal_slack=_cfg_float(scfg, "safe_nominal_slack", 0.12, bucket_name),
                regime_name=bucket_name,
                intervention_budget_rate=(None if not _cfg_has_value(scfg, "intervention_budget_rate", bucket_name) else _cfg_float(scfg, "intervention_budget_rate", 0.20, bucket_name)),
                intervention_budget_used=(None if not _cfg_has_value(scfg, "intervention_budget_used", bucket_name) else _cfg_float(scfg, "intervention_budget_used", 0.0, bucket_name)),
                intervention_budget_steps=(None if not _cfg_has_value(scfg, "intervention_budget_steps", bucket_name) else _cfg_float(scfg, "intervention_budget_steps", 1.0, bucket_name)),
                intervention_budget_penalty=_cfg_float(scfg, "intervention_budget_penalty", 0.25, bucket_name),
                prefer_admitted=_cfg_bool(scfg, "prefer_admitted", False, bucket_name),
                switch_score_margin=_cfg_float(scfg, "switch_score_margin", 0.0, bucket_name),
                safe_switch_score_margin=_cfg_float(scfg, "safe_switch_score_margin", 0.10, bucket_name),
                safe_min_rec_lcb_gain=_cfg_float(scfg, "safe_min_rec_lcb_gain", 0.05, bucket_name),
                safe_min_gap_reduction=_cfg_float(scfg, "safe_min_gap_reduction", 0.15, bucket_name),
                budget_preserve_nominal=_cfg_bool(scfg, "budget_preserve_nominal", True, bucket_name),
                budget_nominal_slack=_cfg_float(scfg, "budget_nominal_slack", 0.08, bucket_name),
                pred_drs=pred_drs,
                pred_direct_value=pred_direct_value,
                pred_direct_rank=pred_direct_rank,
                pred_direct_std=pred_direct_std,
                pred_direct_opportunity=pred_direct_opportunity,
                pred_direct_harm=pred_direct_harm,
                direct_value_certificate=_cfg_bool(scfg, "direct_value_certificate", False, bucket_name),
                direct_value_macro_allowlist=_cfg_value(scfg, "direct_value_macro_allowlist", "brake,yield,merge,stabilize", bucket_name),
                direct_value_lcb_z=_cfg_float(scfg, "direct_value_lcb_z", 1.0, bucket_name),
                direct_value_uncertainty_mode=str(_cfg_value(scfg, "direct_value_uncertainty_mode", "scaled", bucket_name)),
                direct_value_additive_q=_cfg_float(scfg, "direct_value_additive_q", 0.0, bucket_name),
                direct_value_min_nominal_deviation=_cfg_float(scfg, "direct_value_min_nominal_deviation", 0.0, bucket_name),
                direct_value_min_advantage_lcb=_cfg_float(scfg, "direct_value_min_advantage_lcb", 0.035, bucket_name),
                direct_value_min_candidate_value=_cfg_float(scfg, "direct_value_min_candidate_value", 0.45, bucket_name),
                direct_value_max_candidate_std=_cfg_float(scfg, "direct_value_max_candidate_std", 0.35, bucket_name),
                direct_value_max_hard=_cfg_float(scfg, "direct_value_max_hard", gamma_H, bucket_name),
                direct_value_max_harm=_cfg_float(scfg, "direct_value_max_harm", 0.70, bucket_name),
                direct_value_bonus=_cfg_float(scfg, "direct_value_bonus", 0.15, bucket_name),
                direct_value_counts_as_evidence=_cfg_bool(scfg, "direct_value_counts_as_evidence", True, bucket_name),
                direct_value_challenge_nominal=_cfg_bool(scfg, "direct_value_challenge_nominal", True, bucket_name),
                direct_value_max_consecutive=int(_cfg_float(scfg, "direct_value_max_consecutive", 2.0, bucket_name)),
                direct_value_score_mode=_cfg_bool(scfg, "direct_value_score_mode", False, bucket_name),
                direct_value_opportunity_threshold=_cfg_float(scfg, "direct_value_opportunity_threshold", 0.0, bucket_name),
                direct_value_harm_threshold=_cfg_float(scfg, "direct_value_harm_threshold", 1.0, bucket_name),
                direct_value_top1_only=_cfg_bool(scfg, "direct_value_top1_only", False, bucket_name),
                direct_value_policy_first_no_fallback=_cfg_bool(scfg, "direct_value_policy_first_no_fallback", False, bucket_name),
                direct_value_min_rank_margin=_cfg_float(scfg, "direct_value_min_rank_margin", 0.0, bucket_name),
                direct_value_conditional_rank_margin=_cfg_bool(scfg, "direct_value_conditional_rank_margin", False, bucket_name),
                direct_value_risk_controlled_admission=_cfg_bool(scfg, "direct_value_risk_controlled_admission", False, bucket_name),
                deployability_bonus=_cfg_float(scfg, "deployability_bonus", 0.0, bucket_name),
                contact_deployability_bonus=_cfg_float(scfg, "contact_deployability_bonus", 0.0, bucket_name),
                contact_gap_penalty=_cfg_float(scfg, "contact_gap_penalty", 0.0, bucket_name),
                safe_hard_nominal_guard=_cfg_bool(scfg, "safe_hard_nominal_guard", True, bucket_name),
                safe_nominal_max_gap=_cfg_float(scfg, "safe_nominal_max_gap", 0.20, bucket_name),
                safe_override_require_both=_cfg_bool(scfg, "safe_override_require_both", True, bucket_name),
                safe_min_drs_gain=_cfg_float(scfg, "safe_min_drs_gain", 0.10, bucket_name),
                safe_force_nominal_when_feasible=_cfg_bool(scfg, "safe_force_nominal_when_feasible", False, bucket_name),
                safe_force_nominal_mode=str(_cfg_value(scfg, "safe_force_nominal_mode", "feasible", bucket_name)),
                stress_preserve_nominal_min_drs_drop=_cfg_float(scfg, "stress_preserve_nominal_min_drs_drop", -1.0, bucket_name),
                require_admitted_intervention=_cfg_bool(scfg, "require_admitted_intervention", False, bucket_name),
                unadmitted_fallback_to_nominal=_cfg_bool(scfg, "unadmitted_fallback_to_nominal", True, bucket_name),
                intervention_macro_allowlist=_cfg_value(scfg, "intervention_macro_allowlist", "", bucket_name),
                intervention_macro_blocklist=_cfg_value(scfg, "intervention_macro_blocklist", "", bucket_name),
                intervention_require_macro=_cfg_bool(scfg, "intervention_require_macro", False, bucket_name),
                intervention_budget_hard=_cfg_bool(scfg, "intervention_budget_hard", False, bucket_name),
                intervention_budget_hard_min_rec_gain=_cfg_float(scfg, "intervention_budget_hard_min_rec_gain", 0.0, bucket_name),
                intervention_budget_hard_min_drs_gain=_cfg_float(scfg, "intervention_budget_hard_min_drs_gain", 0.0, bucket_name),
                intervention_budget_hard_min_gap_reduction=_cfg_float(scfg, "intervention_budget_hard_min_gap_reduction", 0.0, bucket_name),
                intervention_min_pred_drs=_cfg_float(scfg, "intervention_min_pred_drs", -1.0, bucket_name),
                intervention_max_pred_gap=_cfg_float(scfg, "intervention_max_pred_gap", -1.0, bucket_name),
                require_intervention_evidence=_cfg_bool(scfg, "require_intervention_evidence", False, bucket_name),
                intervention_min_rec_lcb_gain=_cfg_float(scfg, "intervention_min_rec_lcb_gain", 0.0, bucket_name),
                intervention_min_drs_gain=_cfg_float(scfg, "intervention_min_drs_gain", 0.0, bucket_name),
                intervention_min_gap_reduction=_cfg_float(scfg, "intervention_min_gap_reduction", 0.0, bucket_name),
                option_drs_certificate=_cfg_bool(scfg, "option_drs_certificate", False, bucket_name),
                option_drs_certificate_threshold=_cfg_float(scfg, "option_drs_certificate_threshold", 1.01, bucket_name),
                option_drs_certificate_max_gap=_cfg_float(scfg, "option_drs_certificate_max_gap", -1.0, bucket_name),
                option_drs_certificate_rec_slack=_cfg_float(scfg, "option_drs_certificate_rec_slack", 0.0, bucket_name),
                option_drs_certificate_min_rec_lcb=_cfg_float(scfg, "option_drs_certificate_min_rec_lcb", -1.0e9, bucket_name),
                option_drs_certificate_counts_as_evidence=_cfg_bool(scfg, "option_drs_certificate_counts_as_evidence", True, bucket_name),
                relative_recovery_certificate=_cfg_bool(scfg, "relative_recovery_certificate", False, bucket_name),
                relative_recovery_nominal_rec_lcb_max=_cfg_float(scfg, "relative_recovery_nominal_rec_lcb_max", -1.0e9, bucket_name),
                relative_recovery_nominal_gap_min=_cfg_float(scfg, "relative_recovery_nominal_gap_min", 1.0e9, bucket_name),
                relative_recovery_nominal_drs_max=_cfg_float(scfg, "relative_recovery_nominal_drs_max", -1.0, bucket_name),
                relative_recovery_min_rec_gain=_cfg_float(scfg, "relative_recovery_min_rec_gain", 0.10, bucket_name),
                relative_recovery_min_drs=_cfg_float(scfg, "relative_recovery_min_drs", 0.70, bucket_name),
                relative_recovery_min_drs_gain=_cfg_float(scfg, "relative_recovery_min_drs_gain", -1.0, bucket_name),
                relative_recovery_max_drs_drop=_cfg_float(scfg, "relative_recovery_max_drs_drop", -1.0, bucket_name),
                relative_recovery_max_rec_lcb_drop=_cfg_float(scfg, "relative_recovery_max_rec_lcb_drop", -1.0, bucket_name),
                relative_recovery_min_improvement_axes=int(_cfg_float(scfg, "relative_recovery_min_improvement_axes", 1.0, bucket_name)),
                relative_recovery_max_gap=_cfg_float(scfg, "relative_recovery_max_gap", -1.0, bucket_name),
                relative_recovery_max_gap_increase=_cfg_float(scfg, "relative_recovery_max_gap_increase", 0.20, bucket_name),
                relative_recovery_min_gap_reduction=_cfg_float(scfg, "relative_recovery_min_gap_reduction", -1.0, bucket_name),
                relative_recovery_gate=str(_cfg_value(scfg, "relative_recovery_gate", "rec_gain", bucket_name)),
                relative_recovery_use_recovery_pool=_cfg_bool(scfg, "relative_recovery_use_recovery_pool", False, bucket_name),
                recovery_cert_max_hard=_cfg_float(scfg, "recovery_cert_max_hard", 0.0, bucket_name),
                recovery_cert_max_harm=_cfg_float(scfg, "recovery_cert_max_harm", gamma_D, bucket_name),
                relative_recovery_bonus=_cfg_float(scfg, "relative_recovery_bonus", 0.0, bucket_name),
                relative_recovery_counts_as_evidence=_cfg_bool(scfg, "relative_recovery_counts_as_evidence", True, bucket_name),
                candidate_macro_names=candidate_macro_names,
                relative_recovery_macro_allowlist=_cfg_value(scfg, "relative_recovery_macro_allowlist", "", bucket_name),
                relative_recovery_macro_blocklist=_cfg_value(scfg, "relative_recovery_macro_blocklist", "", bucket_name),
                relative_recovery_require_macro=_cfg_bool(scfg, "relative_recovery_require_macro", False, bucket_name),
                relative_recovery_max_intervention_score_gain=_cfg_float(scfg, "relative_recovery_max_intervention_score_gain", -1.0, bucket_name),
                protective_macro_certificate=_cfg_bool(scfg, "protective_macro_certificate", False, bucket_name),
                protective_macro_allowlist=_cfg_value(scfg, "protective_macro_allowlist", "", bucket_name),
                protective_macro_blocklist=_cfg_value(scfg, "protective_macro_blocklist", "", bucket_name),
                protective_macro_nominal_rec_lcb_max=_cfg_float(scfg, "protective_macro_nominal_rec_lcb_max", -1.0e9, bucket_name),
                protective_macro_nominal_gap_min=_cfg_float(scfg, "protective_macro_nominal_gap_min", 1.0e9, bucket_name),
                protective_macro_nominal_drs_max=_cfg_float(scfg, "protective_macro_nominal_drs_max", -1.0, bucket_name),
                protective_macro_min_drs=_cfg_float(scfg, "protective_macro_min_drs", 0.60, bucket_name),
                protective_macro_min_rec_lcb=_cfg_float(scfg, "protective_macro_min_rec_lcb", -1.0e9, bucket_name),
                protective_macro_min_rec_gain=_cfg_float(scfg, "protective_macro_min_rec_gain", -1.0, bucket_name),
                protective_macro_min_gap_reduction=_cfg_float(scfg, "protective_macro_min_gap_reduction", -1.0, bucket_name),
                protective_macro_min_drs_gain=_cfg_float(scfg, "protective_macro_min_drs_gain", -1.0, bucket_name),
                protective_macro_max_drs_drop=_cfg_float(scfg, "protective_macro_max_drs_drop", 0.05, bucket_name),
                protective_macro_max_rec_lcb_drop=_cfg_float(scfg, "protective_macro_max_rec_lcb_drop", 0.10, bucket_name),
                protective_macro_max_gap=_cfg_float(scfg, "protective_macro_max_gap", -1.0, bucket_name),
                protective_macro_max_gap_increase=_cfg_float(scfg, "protective_macro_max_gap_increase", 0.20, bucket_name),
                protective_macro_max_hard=_cfg_float(scfg, "protective_macro_max_hard", gamma_H, bucket_name),
                protective_macro_max_harm=_cfg_float(scfg, "protective_macro_max_harm", gamma_D, bucket_name),
                protective_macro_min_improvement_axes=int(_cfg_float(scfg, "protective_macro_min_improvement_axes", 1.0, bucket_name)),
                protective_macro_gate=str(_cfg_value(scfg, "protective_macro_gate", "axes", bucket_name)),
                protective_macro_score_min_gain=_cfg_float(scfg, "protective_macro_score_min_gain", 0.0, bucket_name),
                protective_macro_score_rec_weight=_cfg_float(scfg, "protective_macro_score_rec_weight", 0.30, bucket_name),
                protective_macro_score_drs_weight=_cfg_float(scfg, "protective_macro_score_drs_weight", 0.55, bucket_name),
                protective_macro_score_gap_weight=_cfg_float(scfg, "protective_macro_score_gap_weight", 0.15, bucket_name),
                protective_macro_bonus=_cfg_float(scfg, "protective_macro_bonus", 0.0, bucket_name),
                protective_macro_counts_as_evidence=_cfg_bool(scfg, "protective_macro_counts_as_evidence", True, bucket_name),
                brake_rescue_certificate=_cfg_bool(scfg, "brake_rescue_certificate", False, bucket_name),
                brake_rescue_macro_name=str(_cfg_value(scfg, "brake_rescue_macro_name", "brake", bucket_name)),
                brake_rescue_min_pred_drs=_cfg_float(scfg, "brake_rescue_min_pred_drs", 0.65, bucket_name),
                brake_rescue_min_pred_r_dep=_cfg_float(scfg, "brake_rescue_min_pred_r_dep", -0.65, bucket_name),
                brake_rescue_min_candidate_gap=_cfg_float(scfg, "brake_rescue_min_candidate_gap", 0.12, bucket_name),
                brake_rescue_max_candidate_gap=_cfg_float(scfg, "brake_rescue_max_candidate_gap", 0.34, bucket_name),
                brake_rescue_max_hard=_cfg_float(scfg, "brake_rescue_max_hard", gamma_H, bucket_name),
                brake_rescue_max_harm=_cfg_float(scfg, "brake_rescue_max_harm", gamma_D, bucket_name),
                brake_rescue_require_nominal_unadmitted=_cfg_bool(scfg, "brake_rescue_require_nominal_unadmitted", True, bucket_name),
                brake_rescue_nominal_rec_lcb_max=_cfg_float(scfg, "brake_rescue_nominal_rec_lcb_max", 0.60, bucket_name),
                brake_rescue_nominal_gap_min=_cfg_float(scfg, "brake_rescue_nominal_gap_min", 0.02, bucket_name),
                brake_rescue_nominal_drs_max=_cfg_float(scfg, "brake_rescue_nominal_drs_max", 1.01, bucket_name),
                brake_rescue_counts_as_evidence=_cfg_bool(scfg, "brake_rescue_counts_as_evidence", True, bucket_name),
                brake_rescue_budget_bypass=_cfg_bool(scfg, "brake_rescue_budget_bypass", True, bucket_name),
                brake_tail_rescue_certificate=_cfg_bool(scfg, "brake_tail_rescue_certificate", False, bucket_name),
                brake_tail_min_pred_drs=_cfg_float(scfg, "brake_tail_min_pred_drs", 0.78, bucket_name),
                brake_tail_min_pred_r_dep=_cfg_float(scfg, "brake_tail_min_pred_r_dep", -0.55, bucket_name),
                brake_tail_min_pred_pcd=_cfg_float(scfg, "brake_tail_min_pred_pcd", 0.24, bucket_name),
                brake_tail_min_candidate_gap=_cfg_float(scfg, "brake_tail_min_candidate_gap", 0.10, bucket_name),
                brake_tail_max_candidate_gap=_cfg_float(scfg, "brake_tail_max_candidate_gap", 0.42, bucket_name),
                brake_tail_high_gap_min=_cfg_float(scfg, "brake_tail_high_gap_min", 0.34, bucket_name),
                brake_tail_low_r_dep_max=_cfg_float(scfg, "brake_tail_low_r_dep_max", 0.00, bucket_name),
                brake_tail_max_hard=_cfg_float(scfg, "brake_tail_max_hard", gamma_H, bucket_name),
                brake_tail_max_harm=_cfg_float(scfg, "brake_tail_max_harm", gamma_D, bucket_name),
                brake_tail_require_nominal_low_headroom=_cfg_bool(scfg, "brake_tail_require_nominal_low_headroom", True, bucket_name),
                brake_tail_nominal_rec_lcb_max=_cfg_float(scfg, "brake_tail_nominal_rec_lcb_max", 0.65, bucket_name),
                brake_tail_nominal_gap_min=_cfg_float(scfg, "brake_tail_nominal_gap_min", 0.02, bucket_name),
                brake_tail_nominal_drs_max=_cfg_float(scfg, "brake_tail_nominal_drs_max", 1.01, bucket_name),
                brake_tail_counts_as_evidence=_cfg_bool(scfg, "brake_tail_counts_as_evidence", True, bucket_name),
                brake_tail_budget_bypass=_cfg_bool(scfg, "brake_tail_budget_bypass", False, bucket_name),
                brake_tail_challenge_budget_bypass=_cfg_bool(scfg, "brake_tail_challenge_budget_bypass", False, bucket_name),
                brake_tail_challenge_cooldown_bypass=_cfg_bool(scfg, "brake_tail_challenge_cooldown_bypass", False, bucket_name),
                brake_tail_challenge_bypass_pcd_gain=_cfg_bool(scfg, "brake_tail_challenge_bypass_pcd_gain", False, bucket_name),
                brake_tail_challenge_min_pred_drs=_cfg_float(scfg, "brake_tail_challenge_min_pred_drs", 0.86, bucket_name),
                brake_tail_challenge_min_pred_pcd=_cfg_float(scfg, "brake_tail_challenge_min_pred_pcd", 0.30, bucket_name),
                brake_tail_challenge_min_candidate_gap=_cfg_float(scfg, "brake_tail_challenge_min_candidate_gap", 0.085, bucket_name),
                brake_tail_challenge_max_candidate_gap=_cfg_float(scfg, "brake_tail_challenge_max_candidate_gap", 0.42, bucket_name),
                brake_tail_challenge_high_gap_min=_cfg_float(scfg, "brake_tail_challenge_high_gap_min", 0.115, bucket_name),
                brake_tail_challenge_low_r_dep_max=_cfg_float(scfg, "brake_tail_challenge_low_r_dep_max", -0.14, bucket_name),
                brake_tail_challenge_max_consecutive=int(_cfg_float(scfg, "brake_tail_challenge_max_consecutive", -1.0, bucket_name)),
                brake_tail_min_nominal_deviation=_cfg_float(scfg, "brake_tail_min_nominal_deviation", 0.0, bucket_name),
                previous_selected_macro=(None if not _cfg_has_value(scfg, "previous_selected_macro", bucket_name) else str(_cfg_value(scfg, "previous_selected_macro", "", bucket_name))),
                same_macro_run_length=(None if not _cfg_has_value(scfg, "same_macro_run_length", bucket_name) else int(_cfg_float(scfg, "same_macro_run_length", 0.0, bucket_name))),
                pcd_rescue_certificate=_cfg_bool(scfg, "pcd_rescue_certificate", False, bucket_name),
                pcd_rescue_macro_allowlist=_cfg_value(scfg, "pcd_rescue_macro_allowlist", "", bucket_name),
                pcd_rescue_macro_blocklist=_cfg_value(scfg, "pcd_rescue_macro_blocklist", "", bucket_name),
                pcd_rescue_min_pred_pcd=_cfg_float(scfg, "pcd_rescue_min_pred_pcd", 0.35, bucket_name),
                pcd_rescue_min_pcd_gain=_cfg_float(scfg, "pcd_rescue_min_pcd_gain", -1.0, bucket_name),
                pcd_rescue_min_pred_drs=_cfg_float(scfg, "pcd_rescue_min_pred_drs", 0.70, bucket_name),
                pcd_rescue_min_pred_r_dep=_cfg_float(scfg, "pcd_rescue_min_pred_r_dep", -0.70, bucket_name),
                pcd_rescue_min_candidate_gap=_cfg_float(scfg, "pcd_rescue_min_candidate_gap", 0.0, bucket_name),
                pcd_rescue_max_candidate_gap=_cfg_float(scfg, "pcd_rescue_max_candidate_gap", 0.50, bucket_name),
                pcd_rescue_max_hard=_cfg_float(scfg, "pcd_rescue_max_hard", gamma_H, bucket_name),
                pcd_rescue_max_harm=_cfg_float(scfg, "pcd_rescue_max_harm", gamma_D, bucket_name),
                pcd_rescue_require_nominal_low_headroom=_cfg_bool(scfg, "pcd_rescue_require_nominal_low_headroom", True, bucket_name),
                pcd_rescue_require_nominal_unadmitted=_cfg_bool(scfg, "pcd_rescue_require_nominal_unadmitted", False, bucket_name),
                pcd_rescue_nominal_rec_lcb_max=_cfg_float(scfg, "pcd_rescue_nominal_rec_lcb_max", 0.65, bucket_name),
                pcd_rescue_nominal_gap_min=_cfg_float(scfg, "pcd_rescue_nominal_gap_min", 0.02, bucket_name),
                pcd_rescue_nominal_drs_max=_cfg_float(scfg, "pcd_rescue_nominal_drs_max", 1.01, bucket_name),
                pcd_rescue_nominal_low_headroom_min_axes=int(_cfg_float(scfg, "pcd_rescue_nominal_low_headroom_min_axes", 1.0, bucket_name)),
                pcd_rescue_max_utility_drop=_cfg_float(scfg, "pcd_rescue_max_utility_drop", -1.0, bucket_name),
                pcd_rescue_large_pcd_gain=_cfg_float(scfg, "pcd_rescue_large_pcd_gain", 0.06, bucket_name),
                pcd_rescue_bonus=_cfg_float(scfg, "pcd_rescue_bonus", 0.0, bucket_name),
                pcd_rescue_counts_as_evidence=_cfg_bool(scfg, "pcd_rescue_counts_as_evidence", True, bucket_name),
                pcd_rescue_budget_bypass=_cfg_bool(scfg, "pcd_rescue_budget_bypass", False, bucket_name),
                safe_cert_min_pred_drs=_cfg_float(scfg, "safe_cert_min_pred_drs", 0.95, bucket_name),
                safe_cert_max_pred_gap=_cfg_float(scfg, "safe_cert_max_pred_gap", 0.20, bucket_name),
                safe_cert_rec_slack=_cfg_float(scfg, "safe_cert_rec_slack", 0.15, bucket_name),
                stress_nominal_anchor=_cfg_bool(scfg, "stress_nominal_anchor", False, bucket_name),
                stress_anchor_drs_floor=_cfg_float(scfg, "stress_anchor_drs_floor", 0.90, bucket_name),
                stress_anchor_max_gap=_cfg_float(scfg, "stress_anchor_max_gap", 0.30, bucket_name),
                stress_anchor_rec_slack=_cfg_float(scfg, "stress_anchor_rec_slack", 0.10, bucket_name),
                stress_anchor_min_drs_gain=_cfg_float(scfg, "stress_anchor_min_drs_gain", 0.06, bucket_name),
                stress_anchor_min_rec_gain=_cfg_float(scfg, "stress_anchor_min_rec_gain", 0.08, bucket_name),
                stress_anchor_min_gap_reduction=_cfg_float(scfg, "stress_anchor_min_gap_reduction", 0.05, bucket_name),
                stress_rescue_challenge_nominal=_cfg_bool(scfg, "stress_rescue_challenge_nominal", False, bucket_name),
                rescue_challenge_min_candidate_gap=_cfg_float(scfg, "rescue_challenge_min_candidate_gap", -1.0, bucket_name),
                rescue_challenge_max_candidate_gap=_cfg_float(scfg, "rescue_challenge_max_candidate_gap", -1.0, bucket_name),
                rescue_challenge_min_pred_drs=_cfg_float(scfg, "rescue_challenge_min_pred_drs", -1.0, bucket_name),
                rescue_challenge_min_pred_pcd=_cfg_float(scfg, "rescue_challenge_min_pred_pcd", -1.0, bucket_name),
                rescue_challenge_min_pcd_gain=_cfg_float(scfg, "rescue_challenge_min_pcd_gain", -1.0, bucket_name),
                rescue_challenge_min_rec_lcb_gain=_cfg_float(scfg, "rescue_challenge_min_rec_lcb_gain", -1.0, bucket_name),
                rescue_challenge_min_drs_gain=_cfg_float(scfg, "rescue_challenge_min_drs_gain", -1.0, bucket_name),
                rescue_challenge_min_gap_reduction=_cfg_float(scfg, "rescue_challenge_min_gap_reduction", -1.0, bucket_name),
                rescue_challenge_min_improvement_axes=int(_cfg_float(scfg, "rescue_challenge_min_improvement_axes", 0.0, bucket_name)),
                rescue_challenge_macro_allowlist=_cfg_value(scfg, "rescue_challenge_macro_allowlist", "", bucket_name),
                rescue_challenge_macro_blocklist=_cfg_value(scfg, "rescue_challenge_macro_blocklist", "", bucket_name),
                rescue_challenge_max_pred_utility=_cfg_float(scfg, "rescue_challenge_max_pred_utility", -1.0, bucket_name),
                rescue_challenge_max_used=int(_cfg_float(scfg, "rescue_challenge_max_used", -1.0, bucket_name)),
                rescue_challenge_score_pcd_weight=_cfg_float(scfg, "rescue_challenge_score_pcd_weight", 1.0, bucket_name),
                rescue_challenge_score_drs_weight=_cfg_float(scfg, "rescue_challenge_score_drs_weight", 0.15, bucket_name),
                rescue_challenge_score_gap_weight=_cfg_float(scfg, "rescue_challenge_score_gap_weight", 0.25, bucket_name),
                rescue_challenge_score_utility_weight=_cfg_float(scfg, "rescue_challenge_score_utility_weight", 0.02, bucket_name),
                rescue_challenge_nominal_guard_min_pcd=_cfg_float(scfg, "rescue_challenge_nominal_guard_min_pcd", -1.0, bucket_name),
                rescue_challenge_nominal_guard_max_gap=_cfg_float(scfg, "rescue_challenge_nominal_guard_max_gap", -1.0, bucket_name),
                intervention_cooldown_steps=int(_cfg_float(scfg, "intervention_cooldown_steps", 0.0, bucket_name)),
                steps_since_last_intervention=(None if not _cfg_has_value(scfg, "steps_since_last_intervention", bucket_name) else _cfg_float(scfg, "steps_since_last_intervention", 1.0e9, bucket_name)),
            )
            gap_arr = np.asarray(pred_gap if pred_gap is not None else np.zeros_like(pred_r_dep), dtype=float)
            score = pred_r_dep - beta * np.maximum(0.0, gap_arr)
            return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, score)
        sel = constrained_lcb_select(
            utility, pred_r_dep, hard, harm, feasible,
            gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D,
            pred_gap=pred_gap, nominal_deviation=nominal_deviation,
            lcb_beta=beta,
            nominal_slack=_cfg_float(scfg, "nominal_slack", 0.03, bucket_name),
            nominal_slack_gap_limit=_cfg_float(scfg, "nominal_slack_gap_limit", 0.50, bucket_name),
            intervention_penalty=_cfg_float(scfg, "intervention_penalty", 0.03, bucket_name),
            deviation_penalty=_cfg_float(scfg, "deviation_penalty", 0.15, bucket_name),
            recovery_bonus=_cfg_float(scfg, "recovery_bonus", 0.02, bucket_name),
            fallback_rec_weight=_cfg_float(scfg, "fallback_rec_weight", 0.10, bucket_name),
            fallback_lcb_margin=_cfg_float(scfg, "fallback_lcb_margin", 0.05, bucket_name),
            fallback_gap_margin=_cfg_float(scfg, "fallback_gap_margin", 0.25, bucket_name),
            nominal_fallback_lcb_slack=_cfg_float(scfg, "nominal_fallback_lcb_slack", 0.05, bucket_name),
        )
        gap_arr = np.asarray(pred_gap if pred_gap is not None else np.zeros_like(pred_r_dep), dtype=float)
        score = pred_r_dep - beta * np.maximum(0.0, gap_arr)
        return BaselineSelection(sel.selected_index, sel.reason, sel.admitted, score)

    if method == "ocrap_teacher":
        sel = crisp_select(utility, teacher_r_dep, hard, harm, feasible, gamma_rec=gamma_rec, gamma_H=gamma_H, gamma_D=gamma_D)
        return BaselineSelection(sel.selected_index, "teacher_deployable_upper_bound", sel.admitted, teacher_r_dep)

    if method == "nominal":
        admitted = np.zeros_like(feasible, dtype=bool)
        idx = 0 if len(feasible) and feasible[0] else _best_by_score(utility, feasible)
        admitted[idx] = True
        return BaselineSelection(idx, "nominal_prefix", admitted, utility)

    if method == "log_replay":
        # Explicit logged/nominal rollout baseline. Candidate generation writes
        # the nominal/log-following prefix as candidate 0 when available; if it is
        # infeasible, fall back to the best feasible utility candidate so the
        # closed-loop runner can continue on degenerate frames.
        admitted = np.zeros_like(feasible, dtype=bool)
        idx = 0 if len(feasible) and feasible[0] else _best_by_score(utility, feasible)
        admitted[idx] = True
        return BaselineSelection(idx, "log_replay_prefix", admitted, utility)

    if method == "idm_proxy":
        # Lightweight IDM-style heuristic over OC-RAP candidates. It does not
        # reimplement a full car-following simulator; it prefers feasible,
        # low-hard-violation and low-harm candidates while keeping utility.
        bcfg = cfg.get("baselines", {}) if isinstance(cfg.get("baselines", {}), dict) else {}
        lam_harm = float(bcfg.get("idm_harm_lambda", 2.0))
        lam_hard = float(bcfg.get("idm_hard_lambda", 15.0))
        score = utility - lam_harm * harm - lam_hard * hard
        idx = _best_by_score(score, feasible)
        admitted = feasible.copy()
        return BaselineSelection(idx, "idm_proxy_utility_safety", admitted, score)

    if method == "mpc_proxy":
        # Lightweight constrained-MPC proxy over the same candidate lattice. The
        # hard safety mask is applied first; inside it the controller maximizes
        # nominal utility. This is a sanity baseline, not a reproduction of a
        # specific published MPC implementation.
        admitted = safe_mask.copy()
        if admitted.any():
            idx = _admit_then_utility(admitted, utility)
        else:
            score = utility - 25.0 * hard - 5.0 * harm
            idx = _best_by_score(score, feasible)
        return BaselineSelection(idx, "mpc_proxy_constrained_lattice", admitted, utility)

    if method == "risk_aware":
        bcfg = cfg.get("baselines", {}) if isinstance(cfg.get("baselines", {}), dict) else {}
        lam_harm = float(bcfg.get("risk_lambda", 1.0))
        lam_hard = float(bcfg.get("hard_lambda", 10.0))
        score = utility - lam_harm * harm - lam_hard * hard
        idx = _best_by_score(score, feasible)
        admitted = feasible.copy()
        return BaselineSelection(idx, "utility_minus_risk", admitted, score)

    if method == "backup_filter":
        # A conventional safety backup admits actions when some branch-wise
        # recovery appears feasible, without enforcing observation consistency.
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        idx = _admit_then_utility(admitted, utility)
        return BaselineSelection(idx, "branchwise_backup_filter", admitted, oracle_signal)

    if method == "oracle_filter":
        # Strong oracle-recoverability baseline: same admission as branch-wise
        # recovery, but scored by oracle recoverability before utility.
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        score = oracle_signal + 1.0e-3 * utility
        if admitted.any():
            idxs = np.where(admitted)[0]
            idx = int(idxs[np.argmax(score[idxs])])
        else:
            idx = _best_by_score(score, feasible)
        return BaselineSelection(idx, "oracle_recoverability_filter", admitted, oracle_signal)

    if method == "contingency":
        # Branch-specific contingency planner: maximize oracle recovery headroom,
        # then nominal utility.  It is expected to fail on oracle artifacts.
        score = oracle_signal + 1.0e-3 * utility
        idx = _best_by_score(score, feasible)
        admitted = safe_mask & (oracle_signal >= gamma_rec)
        return BaselineSelection(idx, "branch_specific_contingency", admitted, score)

    raise ValueError(f"Unknown evaluation method {method!r}; valid methods: {BASELINES}")
