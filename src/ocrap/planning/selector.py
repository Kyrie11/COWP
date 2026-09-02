from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionResult:
    selected_index: int
    reason: str
    admitted: np.ndarray


def crisp_select(utility: np.ndarray, r_dep: np.ndarray, hard: np.ndarray, harm: np.ndarray, feasible: np.ndarray, gamma_rec: float = 0.0, gamma_H: float = 0.0, gamma_D: float = 0.0, nominal_index: int = 0) -> SelectionResult:
    utility = np.asarray(utility, dtype=float)
    r_dep = np.asarray(r_dep, dtype=float)
    hard = np.asarray(hard, dtype=float)
    harm = np.asarray(harm, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    admitted = feasible & (r_dep >= gamma_rec) & (hard <= gamma_H) & (harm <= gamma_D)
    if 0 <= nominal_index < len(utility) and admitted[nominal_index]:
        return SelectionResult(int(nominal_index), "nominal_admitted", admitted)
    if admitted.any():
        idxs = np.where(admitted)[0]
        best = int(idxs[np.argmax(utility[idxs])])
        return SelectionResult(best, "best_admitted_utility", admitted)
    # Lexicographic fallback: minimize hard violation, then harm, then maximize r_dep, then utility.
    order = sorted(range(len(utility)), key=lambda i: (not feasible[i], hard[i], harm[i], -r_dep[i], -utility[i]))
    return SelectionResult(int(order[0]), "lexicographic_fallback", admitted)

def _as_1d_float(x: np.ndarray | None, n: int, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full((n,), float(default), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=float(default))
    return np.where(np.isfinite(arr[:n]), arr[:n], float(default))


def _split_name_set(value) -> set[str]:
    """Parse a comma/space separated macro-name config into a normalized set.

    The selector normally operates on numeric certificates.  v18 showed that
    certificate-only relative recovery can admit `perturb_nominal` and `keep` as
    if they were post-contact recovery maneuvers.  Name sets let the experiment
    explicitly define which non-nominal macro families are allowed to use the
    recovery-feasibility pool, without affecting ordinary scalar admission.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw = []
        for x in value:
            raw.extend(str(x).replace(";", ",").replace("|", ",").split(","))
    else:
        raw = str(value).replace(";", ",").replace("|", ",").replace(" ", ",").split(",")
    return {str(x).strip().lower() for x in raw if str(x).strip()}


def _as_macro_names(candidate_macro_names, n: int) -> list[str]:
    if candidate_macro_names is None:
        return [""] * int(n)
    try:
        vals = list(candidate_macro_names)
    except TypeError:
        vals = [candidate_macro_names]
    out = [str(x).strip().lower() for x in vals[:n]]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def constrained_lcb_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.0,
    nominal_slack: float = 0.0,
    nominal_slack_gap_limit: float = 0.5,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    fallback_rec_weight: float = 0.10,
    fallback_lcb_margin: float = 0.05,
    fallback_gap_margin: float = 0.25,
    nominal_fallback_lcb_slack: float = 0.05,
) -> SelectionResult:
    """Calibrated constrained selector used by OC-RAP in closed loop.

    It uses a conservative recovery lower-bound proxy
    ``r_lcb = r_dep - beta * max(0, oracle_deployable_gap)`` and adds
    explicit intervention/deviation costs so OC-RAP does not over-intervene when
    the nominal prefix is only slightly below the calibrated threshold.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))
    admitted = safe & (rec_lcb >= float(gamma_rec))
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    score = utility - float(intervention_penalty) * intervention - float(deviation_penalty) * dev + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
    if 0 <= nominal_index < n and safe[nominal_index]:
        nom_margin = rec_lcb[nominal_index] - float(gamma_rec)
        nom_gap = gap[nominal_index]
        if admitted[nominal_index]:
            return SelectionResult(int(nominal_index), "nominal_lcb_admitted", admitted)
        if nom_margin >= -float(nominal_slack) and nom_gap <= float(nominal_slack_gap_limit):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_slack_lcb_admitted", admitted)
    if admitted.any():
        cand = np.where(admitted)[0]
        return SelectionResult(int(cand[np.argmax(score[cand])]), "best_admitted_lcb_score", admitted)
    # No candidate satisfies the calibrated recovery constraint.  This branch is
    # deliberately recovery-first.  Earlier versions used a utility-dominated
    # soft fallback, which could pick high-utility prefixes with lower predicted
    # deployable recoverability and larger oracle--deployable gap.  That defeats
    # the purpose of OC-RAP exactly in low-headroom regimes.  We therefore:
    #   1. keep hard/harm safety as the first gate;
    #   2. restrict fallback candidates to the near-best recovery LCB set;
    #   3. within that set, reject candidates with much larger predicted gap;
    #   4. use utility/intervention/deviation only as a final tie-breaker.
    if safe.any():
        fallback_pool = safe.copy()
    elif feasible.any():
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        fallback_pool = near_hard & (harm <= min_harm + 1e-6)
    else:
        fallback_pool = np.ones((n,), dtype=bool)

    cand = np.where(fallback_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_lcb = float(np.max(rec_lcb[cand])) if cand.size else -np.inf

    if 0 <= nominal_index < n and fallback_pool[nominal_index]:
        nom_lcb = float(rec_lcb[nominal_index])
        nom_gap = float(gap[nominal_index])
        best_gap = float(np.min(gap[cand])) if cand.size else nom_gap
        if nom_lcb >= best_lcb - float(nominal_fallback_lcb_slack) and nom_gap <= best_gap + float(fallback_gap_margin):
            return SelectionResult(int(nominal_index), "nominal_recovery_guarded_fallback", admitted)

    near_best_lcb = fallback_pool & (rec_lcb >= best_lcb - float(fallback_lcb_margin))
    cand = np.where(near_best_lcb)[0]
    if cand.size:
        best_gap = float(np.min(gap[cand]))
        gap_guarded = near_best_lcb & (gap <= best_gap + float(fallback_gap_margin))
        if gap_guarded.any():
            cand = np.where(gap_guarded)[0]
    else:
        cand = np.where(fallback_pool)[0]

    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    fallback_score = score - float(fallback_rec_weight) * rec_shortfall - 10.0 * np.maximum(0.0, hard - float(gamma_H)) - 2.0 * np.maximum(0.0, harm - float(gamma_D))
    return SelectionResult(int(cand[np.argmax(fallback_score[cand])]) if cand.size else 0, "recovery_guarded_fallback", admitted)


def calibrated_constrained_select(
    utility: np.ndarray,
    r_dep: np.ndarray,
    hard: np.ndarray,
    harm: np.ndarray,
    feasible: np.ndarray,
    gamma_rec: float = 0.0,
    gamma_H: float = 0.0,
    gamma_D: float = 0.0,
    nominal_index: int = 0,
    *,
    pred_gap: np.ndarray | None = None,
    nominal_deviation: np.ndarray | None = None,
    lcb_beta: float = 0.10,
    shortfall_penalty: float = 1.0,
    gap_penalty: float = 0.05,
    intervention_penalty: float = 0.03,
    deviation_penalty: float = 0.15,
    recovery_bonus: float = 0.02,
    admission_bonus: float = 0.02,
    nominal_slack: float = 0.03,
    nominal_slack_gap_limit: float = 0.50,
    nominal_utility_slack: float = 0.05,
    safe_nominal_slack: float = 0.12,
    regime_name: str | None = None,
    intervention_budget_rate: float | None = None,
    intervention_budget_used: float | None = None,
    intervention_budget_steps: float | None = None,
    intervention_budget_penalty: float = 0.25,
    prefer_admitted: bool = False,
    switch_score_margin: float = 0.0,
    safe_switch_score_margin: float = 0.10,
    safe_min_rec_lcb_gain: float = 0.05,
    safe_min_gap_reduction: float = 0.15,
    budget_preserve_nominal: bool = True,
    budget_nominal_slack: float = 0.08,
    pred_drs: np.ndarray | None = None,
    pred_direct_value: np.ndarray | None = None,
    pred_direct_rank: np.ndarray | None = None,
    pred_direct_std: np.ndarray | None = None,
    pred_direct_opportunity: np.ndarray | None = None,
    pred_direct_harm: np.ndarray | None = None,
    direct_value_certificate: bool = False,
    direct_value_macro_allowlist=None,
    direct_value_lcb_z: float = 1.0,
    direct_value_uncertainty_mode: str = "scaled",
    direct_value_additive_q: float = 0.0,
    direct_value_min_nominal_deviation: float = 0.0,
    direct_value_min_advantage_lcb: float = 0.035,
    direct_value_min_candidate_value: float = 0.45,
    direct_value_max_candidate_std: float = 0.35,
    direct_value_max_hard: float = 0.0,
    direct_value_max_harm: float = 0.70,
    direct_value_bonus: float = 0.15,
    direct_value_counts_as_evidence: bool = True,
    direct_value_challenge_nominal: bool = True,
    direct_value_max_consecutive: int = 2,
    direct_value_score_mode: bool = False,
    direct_value_opportunity_threshold: float = 0.0,
    direct_value_harm_threshold: float = 1.0,
    direct_value_top1_only: bool = False,
    direct_value_policy_first_no_fallback: bool = False,
    direct_value_proposal_top_k: int = 1,
    direct_value_evidence_rerank_top_k: bool = False,
    direct_value_min_rank_margin: float = 0.0,
    direct_value_conditional_rank_margin: bool = False,
    direct_value_risk_controlled_admission: bool = False,
    deployability_bonus: float = 0.0,
    contact_deployability_bonus: float = 0.0,
    contact_gap_penalty: float = 0.0,
    safe_hard_nominal_guard: bool = True,
    safe_nominal_max_gap: float = 0.20,
    safe_override_require_both: bool = True,
    safe_min_drs_gain: float = 0.10,
    safe_force_nominal_when_feasible: bool = False,
    safe_force_nominal_mode: str = "feasible",
    stress_preserve_nominal_min_drs_drop: float = -1.0,
    require_admitted_intervention: bool = False,
    unadmitted_fallback_to_nominal: bool = True,
    intervention_macro_allowlist=None,
    intervention_macro_blocklist=None,
    intervention_require_macro: bool = False,
    intervention_budget_hard: bool = False,
    intervention_budget_hard_min_rec_gain: float = 0.0,
    intervention_budget_hard_min_drs_gain: float = 0.0,
    intervention_budget_hard_min_gap_reduction: float = 0.0,
    intervention_min_pred_drs: float = -1.0,
    intervention_max_pred_gap: float = -1.0,
    safe_cert_min_pred_drs: float = 0.95,
    safe_cert_max_pred_gap: float = 0.20,
    safe_cert_rec_slack: float = 0.15,
    stress_nominal_anchor: bool = False,
    stress_anchor_drs_floor: float = 0.90,
    stress_anchor_max_gap: float = 0.30,
    stress_anchor_rec_slack: float = 0.10,
    stress_anchor_min_drs_gain: float = 0.06,
    stress_anchor_min_rec_gain: float = 0.08,
    stress_anchor_min_gap_reduction: float = 0.05,
    require_intervention_evidence: bool = False,
    intervention_min_rec_lcb_gain: float = 0.00,
    intervention_min_drs_gain: float = 0.00,
    intervention_min_gap_reduction: float = 0.00,
    option_drs_certificate: bool = False,
    option_drs_certificate_threshold: float = 1.01,
    option_drs_certificate_max_gap: float = -1.0,
    option_drs_certificate_rec_slack: float = 0.0,
    option_drs_certificate_min_rec_lcb: float = -1.0e9,
    option_drs_certificate_counts_as_evidence: bool = True,
    relative_recovery_certificate: bool = False,
    relative_recovery_nominal_rec_lcb_max: float = -1.0e9,
    relative_recovery_nominal_gap_min: float = 1.0e9,
    relative_recovery_nominal_drs_max: float = -1.0,
    relative_recovery_min_rec_gain: float = 0.10,
    relative_recovery_min_drs: float = 0.70,
    relative_recovery_min_drs_gain: float = -1.0,
    relative_recovery_max_drs_drop: float = -1.0,
    relative_recovery_max_rec_lcb_drop: float = -1.0,
    relative_recovery_min_improvement_axes: int = 1,
    relative_recovery_max_gap: float = -1.0,
    relative_recovery_max_gap_increase: float = 0.20,
    relative_recovery_min_gap_reduction: float = -1.0,
    relative_recovery_gate: str = "rec_gain",
    relative_recovery_use_recovery_pool: bool = False,
    recovery_cert_max_hard: float = 0.0,
    recovery_cert_max_harm: float | None = None,
    relative_recovery_bonus: float = 0.0,
    relative_recovery_counts_as_evidence: bool = True,
    candidate_macro_names=None,
    relative_recovery_macro_allowlist=None,
    relative_recovery_macro_blocklist=None,
    relative_recovery_require_macro: bool = False,
    relative_recovery_max_intervention_score_gain: float = -1.0,
    protective_macro_certificate: bool = False,
    protective_macro_allowlist=None,
    protective_macro_blocklist=None,
    protective_macro_nominal_rec_lcb_max: float = -1.0e9,
    protective_macro_nominal_gap_min: float = 1.0e9,
    protective_macro_nominal_drs_max: float = -1.0,
    protective_macro_min_drs: float = 0.60,
    protective_macro_min_rec_lcb: float = -1.0e9,
    protective_macro_min_rec_gain: float = -1.0,
    protective_macro_min_gap_reduction: float = -1.0,
    protective_macro_min_drs_gain: float = -1.0,
    protective_macro_max_drs_drop: float = 0.05,
    protective_macro_max_rec_lcb_drop: float = 0.10,
    protective_macro_max_gap: float = -1.0,
    protective_macro_max_gap_increase: float = 0.20,
    protective_macro_max_hard: float = 0.0,
    protective_macro_max_harm: float = 5.0,
    protective_macro_min_improvement_axes: int = 1,
    protective_macro_gate: str = "axes",
    protective_macro_score_min_gain: float = 0.00,
    protective_macro_score_rec_weight: float = 0.30,
    protective_macro_score_drs_weight: float = 0.55,
    protective_macro_score_gap_weight: float = 0.15,
    protective_macro_bonus: float = 0.0,
    protective_macro_counts_as_evidence: bool = True,
    # v23: contact-specific brake rescue channel. This is intentionally not a
    # generic threshold relaxation: it only applies to the physically protective
    # brake macro in contact buckets, after the semantic intervention firewall,
    # and only when nominal itself has failed calibrated admission.  The gate is
    # designed for the v22 failure mode where teacher-audited brake improves PCD
    # but the scalar rec-LCB is under-confident because braking raises the
    # predicted oracle/deployability gap.
    brake_rescue_certificate: bool = False,
    brake_rescue_macro_name: str = "brake",
    brake_rescue_min_pred_drs: float = 0.65,
    brake_rescue_min_pred_r_dep: float = -0.65,
    brake_rescue_min_candidate_gap: float = 0.12,
    brake_rescue_max_candidate_gap: float = 0.34,
    brake_rescue_max_hard: float = 1.0,
    brake_rescue_max_harm: float = 0.70,
    brake_rescue_require_nominal_unadmitted: bool = True,
    brake_rescue_nominal_rec_lcb_max: float = 0.60,
    brake_rescue_nominal_gap_min: float = 0.02,
    brake_rescue_nominal_drs_max: float = 1.01,
    brake_rescue_counts_as_evidence: bool = True,
    brake_rescue_budget_bypass: bool = True,
    # v33: residual/tail brake certificate.  v32 showed that the learned PCD head
    # can still under-rank paper-best brake in contact: brake has high DRS and a
    # plausible absolute PCD, but nominal's over-confident low gap makes the
    # relative PCD-gain gate negative.  This optional certificate admits only
    # brake candidates in a narrow residual-shape band: either a high predicted
    # oracle-deployable gap (tail uncertainty) or low learned R_dep with high DRS.
    # It does not use teacher/audit labels at inference time.
    brake_tail_rescue_certificate: bool = False,
    brake_tail_min_pred_drs: float = 0.78,
    brake_tail_min_pred_r_dep: float = -0.55,
    brake_tail_min_pred_pcd: float = 0.24,
    brake_tail_min_candidate_gap: float = 0.10,
    brake_tail_max_candidate_gap: float = 0.42,
    brake_tail_high_gap_min: float = 0.34,
    brake_tail_low_r_dep_max: float = 0.00,
    brake_tail_max_hard: float = 1.0,
    brake_tail_max_harm: float = 0.70,
    brake_tail_require_nominal_low_headroom: bool = True,
    brake_tail_nominal_rec_lcb_max: float = 0.65,
    brake_tail_nominal_gap_min: float = 0.02,
    brake_tail_nominal_drs_max: float = 1.01,
    brake_tail_counts_as_evidence: bool = True,
    brake_tail_budget_bypass: bool = False,
    # v36: permit only strict residual-tail challenges to override the hard
    # exposure budget; broad brake rescue remains budget-limited.
    brake_tail_challenge_budget_bypass: bool = False,
    # v38: allow only the strict residual-tail challenge subset to continue
    # through the ordinary cooldown. This prevents wasting a one-step brake
    # just before an audited low-headroom state while preserving cooldown for
    # broad brake-rescue and generic recovery macros.
    brake_tail_challenge_cooldown_bypass: bool = False,
    # v34: when a residual brake-tail candidate has already passed the absolute
    # certificate, let it challenge an admitted nominal even if learned relative
    # PCD gain is negative. This is deliberately contact-only via config and
    # targets the remaining v33 failure mode: nominal has over-confident low gap
    # while paper-best brake has high absolute recoverability.
    brake_tail_challenge_bypass_pcd_gain: bool = False,
    # v35: split the residual-tail *certificate* from the more aggressive
    # admitted-nominal *challenge bypass*.  v34 proved the bypass can clear the
    # contact brake tail, but it also over-fired because every broad tail
    # certificate could bypass the relative learned-PCD gain.  These stricter
    # challenge-only gates keep the broad certificate available for ordinary
    # admission while allowing PCD-gain bypass only for high-confidence tail
    # shapes: high DRS/PCD plus either a material predicted gap or clearly low
    # learned R_dep.
    brake_tail_challenge_min_pred_drs: float = 0.86,
    brake_tail_challenge_min_pred_pcd: float = 0.30,
    brake_tail_challenge_min_candidate_gap: float = 0.085,
    brake_tail_challenge_max_candidate_gap: float = 0.42,
    brake_tail_challenge_high_gap_min: float = 0.115,
    brake_tail_challenge_low_r_dep_max: float = -0.14,
    # v39: bound repeated cooldown-bypass brake decisions. Consecutive brake
    # controls are one recovery episode, not independent tail discoveries.
    brake_tail_challenge_max_consecutive: int = -1,
    brake_tail_min_nominal_deviation: float = 0.0,
    previous_selected_macro: str | None = None,
    same_macro_run_length: int | None = None,
    # v24: Budgeted Macro-Rescue Certificate (BMRC).  This generalizes the
    # v23 brake rescue from a fixed macro threshold into a predicted
    # post-contact-deployability admission test.  It can be enabled per bucket
    # and per macro family, and it is explicitly budget-aware so repeated
    # interventions cannot silently consume nominal utility.
    pcd_rescue_certificate: bool = False,
    pcd_rescue_macro_allowlist=None,
    pcd_rescue_macro_blocklist=None,
    pcd_rescue_min_pred_pcd: float = 0.35,
    pcd_rescue_min_pcd_gain: float = -1.0,
    pcd_rescue_min_pred_drs: float = 0.70,
    pcd_rescue_min_pred_r_dep: float = -0.70,
    pcd_rescue_min_candidate_gap: float = 0.0,
    pcd_rescue_max_candidate_gap: float = 0.50,
    pcd_rescue_max_hard: float = 1.0,
    pcd_rescue_max_harm: float = 0.70,
    pcd_rescue_require_nominal_low_headroom: bool = True,
    pcd_rescue_require_nominal_unadmitted: bool = False,
    pcd_rescue_nominal_rec_lcb_max: float = 0.65,
    pcd_rescue_nominal_gap_min: float = 0.02,
    pcd_rescue_nominal_drs_max: float = 1.01,
    pcd_rescue_nominal_low_headroom_min_axes: int = 1,
    pcd_rescue_max_utility_drop: float = -1.0,
    pcd_rescue_large_pcd_gain: float = 0.06,
    pcd_rescue_bonus: float = 0.0,
    pcd_rescue_counts_as_evidence: bool = True,
    pcd_rescue_budget_bypass: bool = False,
    # v25: let a certified recovery macro challenge an already admitted nominal
    # prefix in stress buckets.  v24 fixed missing brake candidates, but the
    # nominal-preserving return still short-circuited the selector whenever the
    # learned nominal LCB was over-confident.  This switch is intentionally
    # disabled by default and should only be enabled for near/contact buckets.
    stress_rescue_challenge_nominal: bool = False,
    # v26: guarded nominal challenge.  v25 showed that merely allowing every
    # rescue-certified brake/PCD candidate to challenge nominal over-selected
    # high-utility but non-deployable brakes in contact.  These guards define a
    # second, stricter challenge frontier that is applied only when nominal is
    # already calibrated-admitted.  The ordinary rescue certificates are still
    # available for the easier case where nominal is not admitted.
    rescue_challenge_min_candidate_gap: float = -1.0,
    rescue_challenge_max_candidate_gap: float = -1.0,
    rescue_challenge_min_pred_drs: float = -1.0,
    rescue_challenge_min_pred_pcd: float = -1.0,
    rescue_challenge_min_pcd_gain: float = -1.0,
    rescue_challenge_min_rec_lcb_gain: float = -1.0,
    rescue_challenge_min_drs_gain: float = -1.0,
    rescue_challenge_min_gap_reduction: float = -1.0,
    rescue_challenge_min_improvement_axes: int = 0,
    rescue_challenge_macro_allowlist=None,
    rescue_challenge_macro_blocklist=None,
    rescue_challenge_max_pred_utility: float = -1.0,
    rescue_challenge_max_used: int = -1,
    rescue_challenge_score_pcd_weight: float = 1.0,
    rescue_challenge_score_drs_weight: float = 0.15,
    rescue_challenge_score_gap_weight: float = 0.25,
    rescue_challenge_score_utility_weight: float = 0.02,
    rescue_challenge_nominal_guard_min_pcd: float = -1.0,
    rescue_challenge_nominal_guard_max_gap: float = -1.0,
    # Closed-loop exposure control: minimum number of decisions since the last
    # non-nominal action before another recovery prefix may be executed.  This
    # is stronger than a soft rate penalty and prevents repeated braking bursts.
    intervention_cooldown_steps: int = 0,
    steps_since_last_intervention: float | None = None,
) -> SelectionResult:
    """Soft calibrated OC-RAP selector.

    ``constrained_lcb_select`` is deliberately hard-gated: candidates below the
    calibrated recovery threshold all fall into a recovery-first fallback.  That
    is useful for stress tests, but in natural/safe scenes it can over-intervene
    because a candidate that misses gamma by a tiny amount is treated like one
    that misses it by a large amount.  This selector keeps hard/harm feasibility
    as a constraint, then turns recovery-threshold shortfall into a continuous
    penalty.  It also has an optional intervention budget and a stronger
    nominal-preservation rule for safe/background buckets.
    """
    utility = np.asarray(utility, dtype=float).reshape(-1)
    n = int(utility.size)
    if n <= 0:
        return SelectionResult(0, "empty_calibrated_selector", np.zeros((0,), dtype=bool))
    r_dep = _as_1d_float(r_dep, n, default=-np.inf)
    hard = _as_1d_float(hard, n, default=np.inf)
    harm = _as_1d_float(harm, n, default=np.inf)
    feasible = np.asarray(feasible, dtype=bool).reshape(-1)
    if feasible.size < n:
        feasible = np.pad(feasible, (0, n - feasible.size), constant_values=False)
    feasible = feasible[:n]
    gap = np.maximum(0.0, _as_1d_float(pred_gap, n, default=0.0))
    dev = np.maximum(0.0, _as_1d_float(nominal_deviation, n, default=0.0))
    drs_proxy = np.clip(_as_1d_float(pred_drs, n, default=0.0), 0.0, 1.0)
    direct_value = _as_1d_float(pred_direct_value, n, default=0.0)
    if not bool(direct_value_score_mode):
        direct_value = np.clip(direct_value, 0.0, 1.0)
    direct_rank = _as_1d_float(pred_direct_rank, n, default=0.0) if pred_direct_rank is not None else direct_value.copy()
    direct_std = np.maximum(0.0, _as_1d_float(pred_direct_std, n, default=1.0))
    # Backward-compatible default 1.0 keeps v40-v43 checkpoints usable when the
    # v44 opportunity gate is disabled (threshold=0).
    direct_opportunity = np.clip(_as_1d_float(pred_direct_opportunity, n, default=1.0), 0.0, 1.0)
    direct_pred_harm = np.clip(_as_1d_float(pred_direct_harm, n, default=0.0), 0.0, 1.0)
    macro_names = _as_macro_names(candidate_macro_names, n)

    # v22: separate candidate-family coverage from recovery-maneuver semantics.
    # Scalar/option certificates are purely numerical, so without a global
    # intervention macro gate they can still execute perturb_nominal, lane_shift,
    # or other exploration families as if they were paper-valid recoveries.
    # Nominal is always allowed; non-nominal interventions must satisfy this
    # semantic mask whenever configured.
    intervention_macro_allow = _split_name_set(intervention_macro_allowlist)
    intervention_macro_block = _split_name_set(intervention_macro_blocklist)
    intervention_macro_mask = np.ones((n,), dtype=bool)
    if intervention_macro_allow:
        intervention_macro_mask &= np.asarray([m in intervention_macro_allow for m in macro_names], dtype=bool)
    if intervention_macro_block:
        intervention_macro_mask &= ~np.asarray([m in intervention_macro_block for m in macro_names], dtype=bool)
    if bool(intervention_require_macro):
        intervention_macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)
    if 0 <= int(nominal_index) < n:
        intervention_macro_mask[int(nominal_index)] = True

    rec_lcb = r_dep - float(lcb_beta) * gap
    safe = feasible & (hard <= float(gamma_H)) & (harm <= float(gamma_D))

    # CRISP's nominal admission uses hard-rule/harm feasibility.  In near-contact
    # and post-contact regimes, however, the *recovery* prefix may deliberately
    # violate a nominal lane/comfort rule while reducing deployable recovery loss
    # (e.g. stabilize after contact, pull over, or brake/yield into a low-utility
    # state).  Keep scalar admission strict, but optionally allow relative
    # recovery certificates to use a regime-specific recovery-feasibility pool.
    recovery_hard_limit = float(gamma_H)
    if bool(relative_recovery_use_recovery_pool):
        recovery_hard_limit = max(float(gamma_H), float(recovery_cert_max_hard))
    recovery_harm_limit = float(gamma_D) if recovery_cert_max_harm is None else float(recovery_cert_max_harm)
    recovery_safe = feasible & (hard <= recovery_hard_limit) & (harm <= recovery_harm_limit)

    scalar_admitted = safe & (rec_lcb >= float(gamma_rec))

    # v15 dual certificate.  The scalar OC-MERO LCB can be overly pessimistic
    # under the closed-loop distribution shift observed in v14, while the
    # observation-consistent shared-option head can still identify a deployable
    # recovery option.  A non-nominal prefix may therefore be certified either by
    # the calibrated scalar recovery LCB, or by a directly predicted shared
    # recovery-option success certificate with gap/LCB guards.  This keeps the
    # paper claim intact: interventions still require an observation-consistent
    # deployability certificate; they are no longer blocked solely because the
    # aggregate scalar head is conservative.
    option_certified = np.zeros((n,), dtype=bool)
    if bool(option_drs_certificate):
        option_certified = safe & (drs_proxy >= float(option_drs_certificate_threshold))
        if float(option_drs_certificate_max_gap) >= 0.0:
            option_certified &= gap <= float(option_drs_certificate_max_gap)
        option_certified &= rec_lcb >= float(gamma_rec) - float(option_drs_certificate_rec_slack)
        option_certified &= rec_lcb >= float(option_drs_certificate_min_rec_lcb)

    # v16 relative-recovery certificate.  The v15 result showed that a single
    # absolute scalar/DRS threshold can still abstain throughout contact scenes
    # where nominal is clearly low-headroom but one of the current candidates is
    # relatively better.  This certificate is deliberately counterfactual: it is
    # enabled only in a nominal-opportunity state, and it admits non-nominal
    # prefixes that improve predicted deployable recovery over nominal while
    # satisfying DRS and gap guards.  This keeps the method certifiable without
    # returning to the v13 soft fallback.
    relative_certified = np.zeros((n,), dtype=bool)
    protective_certified = np.zeros((n,), dtype=bool)
    brake_rescue_certified = np.zeros((n,), dtype=bool)
    brake_tail_rescue_certified = np.zeros((n,), dtype=bool)
    pcd_rescue_certified = np.zeros((n,), dtype=bool)
    pcd_proxy = np.clip(drs_proxy, 0.0, 1.0) * (1.0 / (1.0 + np.exp(-np.clip(r_dep, -40.0, 40.0)))) * np.exp(-np.clip(gap, 0.0, 20.0))
    pcd_gain_vs_nom = np.zeros((n,), dtype=float)
    rec_gain_vs_nom = np.zeros((n,), dtype=float)
    drs_gain_vs_nom = np.zeros((n,), dtype=float)
    gap_reduction_vs_nom = np.zeros((n,), dtype=float)
    if 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        rec_gain_vs_nom = rec_lcb - rec_lcb[ni]
        drs_gain_vs_nom = drs_proxy - drs_proxy[ni]
        gap_reduction_vs_nom = gap[ni] - gap
        pcd_gain_vs_nom = pcd_proxy - pcd_proxy[ni]

    # v40 OC-UVRA: uncertainty-aware counterfactual value is a preference
    # certificate, never a replacement for OC-MERO feasibility/admission.  It
    # may challenge nominal only when its lower-confidence advantage is positive
    # and the candidate is physically/semantically bounded.
    direct_advantage_lcb = np.full((n,), -np.inf, dtype=float)
    direct_value_challenge = np.zeros((n,), dtype=bool)
    direct_actionable = np.zeros((n,), dtype=bool)
    if bool(direct_value_certificate) and 0 <= int(nominal_index) < n and pred_direct_value is not None:
        ni = int(nominal_index)
        raw_direct_advantage = direct_value - direct_value[ni]
        uncertainty_mode = str(direct_value_uncertainty_mode or "scaled").strip().lower()
        if uncertainty_mode in {"additive", "conformal_additive", "residual"}:
            # v41: q is calibrated on max candidate over-estimation within the
            # deterministic actionable candidate set.  This remains valid after
            # selecting the best candidate and does not trust self-reported std.
            direct_advantage_lcb = raw_direct_advantage - float(direct_value_additive_q)
        elif uncertainty_mode in {"none", "raw", "risk_selective", "selective", "risk_controlled"}:
            # v43 OC-RSC uses a deterministic score threshold fitted and
            # verified on disjoint scene-time folds. The threshold is applied
            # through direct_value_min_advantage_lcb below.
            direct_advantage_lcb = raw_direct_advantage
        else:
            pair_std = np.sqrt(np.maximum(0.0, direct_std * direct_std + direct_std[ni] * direct_std[ni]))
            direct_advantage_lcb = raw_direct_advantage - float(direct_value_lcb_z) * pair_std
        direct_macro_allow = _split_name_set(direct_value_macro_allowlist)
        direct_macro_mask = np.asarray([bool(m) for m in macro_names], dtype=bool)
        if direct_macro_allow:
            direct_macro_mask &= np.asarray([m in direct_macro_allow for m in macro_names], dtype=bool)
        candidate_floor_ok = np.ones((n,), dtype=bool) if bool(direct_value_score_mode) else (direct_value >= float(direct_value_min_candidate_value))
        physical_direct = (
            feasible
            & (hard <= float(direct_value_max_hard))
            & (harm <= float(direct_value_max_harm))
            & direct_macro_mask
            & (dev >= float(direct_value_min_nominal_deviation))
            & candidate_floor_ok
            & ((direct_std <= float(direct_value_max_candidate_std)) if uncertainty_mode not in {"additive", "conformal_additive", "residual", "none", "raw", "risk_selective", "selective", "risk_controlled"} else np.ones((n,), dtype=bool))
        )
        physical_direct[ni] = False
        evidence_ok = (
            (direct_opportunity >= float(direct_value_opportunity_threshold))
            & (direct_pred_harm <= float(direct_value_harm_threshold))
        )
        direct_actionable = physical_direct & evidence_ok
        if bool(direct_value_top1_only):
            raw_rank_advantage = direct_rank - direct_rank[ni]
            if bool(direct_value_evidence_rerank_top_k):
                # v48.13 TERRA: freeze a preference top-k proposal first, then
                # rerank only certified proposal members by ordinal evidence.
                physical_idx = np.where(physical_direct)[0]
                proposal_k = min(max(1, int(direct_value_proposal_top_k)), int(physical_idx.size))
                keep = np.zeros((n,), dtype=bool)
                if proposal_k > 0:
                    ordered = physical_idx[np.argsort(-raw_rank_advantage[physical_idx], kind="stable")][:proposal_k]
                    eligible = ordered[evidence_ok[ordered]]
                    if eligible.size:
                        evidence_score = raw_direct_advantage
                        chosen = int(eligible[np.argmax(evidence_score[eligible])])
                        alternatives = [float(evidence_score[j]) for j in eligible if int(j) != chosen]
                        second_best = max(alternatives) if alternatives else float(evidence_score[chosen] - 1.0)
                        evidence_margin = float(evidence_score[chosen] - second_best)
                        if float(direct_value_min_rank_margin) <= 0.0 or evidence_margin >= float(direct_value_min_rank_margin):
                            keep[chosen] = True
                direct_actionable = keep
            else:
                rank_pool = physical_direct if bool(direct_value_policy_first_no_fallback) else direct_actionable
                if bool(rank_pool.any()):
                    chosen = int(np.argmax(np.where(rank_pool, raw_rank_advantage, -np.inf)))
                    alternatives = [float(raw_rank_advantage[j]) for j in np.where(rank_pool)[0] if int(j) != chosen]
                    if not bool(direct_value_conditional_rank_margin):
                        alternatives.append(0.0)
                    second_best = max(alternatives) if alternatives else float(raw_rank_advantage[chosen] - 1.0)
                    rank_margin = float(raw_rank_advantage[chosen] - second_best)
                    certified = bool(evidence_ok[chosen])
                    certified &= float(direct_value_min_rank_margin) <= 0.0 or rank_margin >= float(direct_value_min_rank_margin)
                    keep = np.zeros((n,), dtype=bool)
                    if certified:
                        keep[chosen] = True
                    direct_actionable = keep
                else:
                    direct_actionable[:] = False
        direct_value_challenge = direct_actionable & (direct_advantage_lcb >= float(direct_value_min_advantage_lcb))
        if int(direct_value_max_consecutive) >= 0:
            prev_macro = str(previous_selected_macro or "").strip().lower()
            try:
                prev_run = int(same_macro_run_length or 0)
            except Exception:
                prev_run = 0
            if prev_run >= int(direct_value_max_consecutive):
                direct_value_challenge &= np.asarray([m != prev_macro for m in macro_names], dtype=bool)

    if bool(relative_recovery_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        nominal_opportunity = (
            rec_lcb[ni] <= float(relative_recovery_nominal_rec_lcb_max)
            or gap[ni] >= float(relative_recovery_nominal_gap_min)
            or (float(relative_recovery_nominal_drs_max) >= 0.0 and drs_proxy[ni] <= float(relative_recovery_nominal_drs_max))
        )
        if bool(nominal_opportunity):
            # v17: use a dominance-style relative certificate instead of only an
            # absolute rec-LCB gain.  The v16 audits showed that nominal can be
            # falsely overconfident in contact scenes: recoverable alternatives
            # exist, but many are blocked because the scalar head is conservative
            # or because the maneuver has a nominal hard-rule flag.  A relative
            # recovery certificate is valid when it is observation-consistent,
            # has enough predicted shared-option success, does not create an
            # excessive oracle--deployability gap, and improves at least one
            # deployability axis over nominal (recovery LCB, DRS proxy, or gap).
            base_pool = recovery_safe if bool(relative_recovery_use_recovery_pool) else safe
            rec_ok = rec_gain_vs_nom >= float(relative_recovery_min_rec_gain)
            drs_gain_ok = np.zeros((n,), dtype=bool)
            if float(relative_recovery_min_drs_gain) >= 0.0:
                drs_gain_ok = drs_gain_vs_nom >= float(relative_recovery_min_drs_gain)
            gap_ok_gain = np.zeros((n,), dtype=bool)
            if float(relative_recovery_min_gap_reduction) >= 0.0:
                gap_ok_gain = gap_reduction_vs_nom >= float(relative_recovery_min_gap_reduction)
            gate = str(relative_recovery_gate or "rec_gain").strip().lower()
            # v18: Pareto lower-envelope certificate.  v17's rec_or_gap gate
            # improved contact audits, but it could certify a recovery prefix that
            # merely reduced predicted gap while lowering the shared deployability
            # proxy.  That created offline false admissions and one bad contact
            # pull-over.  The certificate below separates *which axis shows
            # opportunity* from *which axes must not deteriorate*: a candidate may
            # be admitted by recovery, DRS, or gap improvement, but configurable
            # non-inferiority guards keep it on the deployability Pareto frontier
            # relative to nominal.
            improve_count = rec_ok.astype(int) + drs_gain_ok.astype(int) + gap_ok_gain.astype(int)
            if gate in {"any", "any_gain", "dominance", "or"}:
                improvement_ok = improve_count >= max(1, int(relative_recovery_min_improvement_axes))
            elif gate in {"two", "two_of_three", "2of3"}:
                improvement_ok = improve_count >= max(2, int(relative_recovery_min_improvement_axes))
            elif gate in {"rec_or_gap", "headroom_or_gap"}:
                improvement_ok = rec_ok | gap_ok_gain
            elif gate in {"drs", "drs_gain", "shared_option"}:
                improvement_ok = drs_gain_ok
            elif gate in {"drs_or_gap", "shared_or_gap"}:
                improvement_ok = drs_gain_ok | gap_ok_gain
            elif gate in {"pareto", "pareto_lcb", "pareto_frontier"}:
                improvement_ok = improve_count >= max(1, int(relative_recovery_min_improvement_axes))
            else:
                improvement_ok = rec_ok
            macro_allow = _split_name_set(relative_recovery_macro_allowlist)
            macro_block = _split_name_set(relative_recovery_macro_blocklist)
            macro_mask = np.ones((n,), dtype=bool)
            if macro_allow:
                macro_mask &= np.asarray([m in macro_allow for m in macro_names], dtype=bool)
            if macro_block:
                macro_mask &= ~np.asarray([m in macro_block for m in macro_names], dtype=bool)
            if bool(relative_recovery_require_macro):
                macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)

            relative_certified = base_pool & improvement_ok & macro_mask
            relative_certified &= drs_proxy >= float(relative_recovery_min_drs)
            # DRS non-inferiority can now be enforced for every gate.  Setting the
            # gain threshold negative preserves the old behavior.
            if float(relative_recovery_min_drs_gain) >= 0.0:
                relative_certified &= drs_gain_vs_nom >= float(relative_recovery_min_drs_gain)
            if float(relative_recovery_max_drs_drop) >= 0.0:
                relative_certified &= drs_gain_vs_nom >= -float(relative_recovery_max_drs_drop)
            if float(relative_recovery_max_rec_lcb_drop) >= 0.0:
                relative_certified &= rec_gain_vs_nom >= -float(relative_recovery_max_rec_lcb_drop)
            if float(relative_recovery_max_gap) >= 0.0:
                relative_certified &= gap <= float(relative_recovery_max_gap)
            if float(relative_recovery_max_gap_increase) >= 0.0:
                relative_certified &= gap <= gap[ni] + float(relative_recovery_max_gap_increase)
            # v19: relative recovery is not an aggressive trigger.  When enabled,
            # reject candidates whose final learned score advantage is so large
            # that the decision is effectively a utility/deployability bonus rather
            # than a recovery-certificate decision.  The score itself is computed
            # below, so this guard is applied later after score construction.
            relative_certified[ni] = False

    # v20 protective-macro certificate.  v19 correctly blocked nominal-like
    # perturbations, but it also became too conservative and kept missing contact
    # cases whose audited best candidate was a brake/stabilize maneuver.  This
    # certificate is a second, narrower channel: only explicitly protective macro
    # families may use it, and they must be in a low-headroom nominal state while
    # satisfying non-inferiority guards.  It does not certify generic lane/utility
    # changes and therefore keeps the paper claim that interventions are semantic
    # recovery maneuvers, not aggressive trigger actions.
    if bool(protective_macro_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        protective_opportunity = (
            rec_lcb[ni] <= float(protective_macro_nominal_rec_lcb_max)
            or gap[ni] >= float(protective_macro_nominal_gap_min)
            or (float(protective_macro_nominal_drs_max) >= 0.0 and drs_proxy[ni] <= float(protective_macro_nominal_drs_max))
        )
        if bool(protective_opportunity):
            macro_allow = _split_name_set(protective_macro_allowlist)
            macro_block = _split_name_set(protective_macro_blocklist)
            macro_mask = np.ones((n,), dtype=bool)
            if macro_allow:
                macro_mask &= np.asarray([m in macro_allow for m in macro_names], dtype=bool)
            if macro_block:
                macro_mask &= ~np.asarray([m in macro_block for m in macro_names], dtype=bool)
            # An empty macro name should never pass this semantic certificate.
            macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)

            protective_pool = feasible & (hard <= float(protective_macro_max_hard)) & (harm <= float(protective_macro_max_harm))
            rec_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_rec_gain) >= 0.0:
                rec_ok = rec_gain_vs_nom >= float(protective_macro_min_rec_gain)
            drs_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_drs_gain) >= 0.0:
                drs_ok = drs_gain_vs_nom >= float(protective_macro_min_drs_gain)
            gap_ok = np.zeros((n,), dtype=bool)
            if float(protective_macro_min_gap_reduction) >= 0.0:
                gap_ok = gap_reduction_vs_nom >= float(protective_macro_min_gap_reduction)
            improve_count = rec_ok.astype(int) + drs_ok.astype(int) + gap_ok.astype(int)
            # v21: support a calibrated deployability-vector gate for protective
            # macros.  Brake/stabilize can legitimately trade geometric gap for
            # shared-option robustness immediately after contact, so requiring a
            # pure gap reduction (v20) can suppress exactly the macro we want to
            # test.  The score gate still requires semantic macro eligibility,
            # hard/harm feasibility, absolute DRS, and non-inferiority guards.
            protective_score_gain = (
                float(protective_macro_score_rec_weight) * rec_gain_vs_nom
                + float(protective_macro_score_drs_weight) * drs_gain_vs_nom
                + float(protective_macro_score_gap_weight) * gap_reduction_vs_nom
            )
            gate = str(protective_macro_gate or "axes").lower()
            if gate == "score":
                improvement_ok = protective_score_gain >= float(protective_macro_score_min_gain)
            elif gate == "drs_or_score":
                improvement_ok = drs_ok | (protective_score_gain >= float(protective_macro_score_min_gain))
            elif gate == "any":
                improvement_ok = rec_ok | drs_ok | gap_ok
            else:
                improvement_ok = improve_count >= max(1, int(protective_macro_min_improvement_axes))

            protective_certified = protective_pool & macro_mask & improvement_ok
            protective_certified &= drs_proxy >= float(protective_macro_min_drs)
            protective_certified &= rec_lcb >= float(protective_macro_min_rec_lcb)
            if float(protective_macro_max_drs_drop) >= 0.0:
                protective_certified &= drs_gain_vs_nom >= -float(protective_macro_max_drs_drop)
            if float(protective_macro_max_rec_lcb_drop) >= 0.0:
                protective_certified &= rec_gain_vs_nom >= -float(protective_macro_max_rec_lcb_drop)
            if float(protective_macro_max_gap) >= 0.0:
                protective_certified &= gap <= float(protective_macro_max_gap)
            if float(protective_macro_max_gap_increase) >= 0.0:
                protective_certified &= gap <= gap[ni] + float(protective_macro_max_gap_increase)
            protective_certified[ni] = False

    # v23 brake rescue certificate.  Unlike v21/v22 protective_macro_score, this
    # does not trust the scalar recovery LCB to rank contact braking.  It uses a
    # macro-specific physical prior plus calibrated shared-option evidence: a
    # brake rescue is admissible only in contact-like low-headroom states, only
    # if nominal is not already calibrated-admitted, and only when the brake
    # candidate lies in a moderate uncertainty band (large enough gap to indicate
    # recovery ambiguity, not so large that it is an unbounded artifact).
    if bool(brake_rescue_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        brake_name = str(brake_rescue_macro_name or "brake").strip().lower()
        macro_arr = np.asarray([str(m).strip().lower() for m in macro_names], dtype=object)
        nominal_gate = (
            rec_lcb[ni] <= float(brake_rescue_nominal_rec_lcb_max)
            or gap[ni] >= float(brake_rescue_nominal_gap_min)
            or drs_proxy[ni] <= float(brake_rescue_nominal_drs_max)
        )
        if bool(brake_rescue_require_nominal_unadmitted):
            nominal_gate = nominal_gate and (not bool(scalar_admitted[ni] or option_certified[ni] or relative_certified[ni] or protective_certified[ni]))
        brake_rescue_certified = (
            feasible
            & (hard <= float(brake_rescue_max_hard))
            & (harm <= float(brake_rescue_max_harm))
            & (macro_arr == brake_name)
            & (drs_proxy >= float(brake_rescue_min_pred_drs))
            & (r_dep >= float(brake_rescue_min_pred_r_dep))
            & (gap >= float(brake_rescue_min_candidate_gap))
            & (gap <= float(brake_rescue_max_candidate_gap))
            & bool(nominal_gate)
        )
        # Keep the global semantic firewall as the final arbiter.
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            brake_rescue_certified &= intervention_macro_mask

    if bool(brake_tail_rescue_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        brake_name = str(brake_rescue_macro_name or "brake").strip().lower()
        macro_arr = np.asarray([str(m).strip().lower() for m in macro_names], dtype=object)
        nominal_gate = True
        if bool(brake_tail_require_nominal_low_headroom):
            nominal_gate = (
                rec_lcb[ni] <= float(brake_tail_nominal_rec_lcb_max)
                or gap[ni] >= float(brake_tail_nominal_gap_min)
                or drs_proxy[ni] <= float(brake_tail_nominal_drs_max)
            )
        tail_shape = (gap >= float(brake_tail_high_gap_min)) | (r_dep <= float(brake_tail_low_r_dep_max))
        brake_tail_rescue_certified = (
            feasible
            & (hard <= float(brake_tail_max_hard))
            & (harm <= float(brake_tail_max_harm))
            & (macro_arr == brake_name)
            & (drs_proxy >= float(brake_tail_min_pred_drs))
            & (r_dep >= float(brake_tail_min_pred_r_dep))
            & (pcd_proxy >= float(brake_tail_min_pred_pcd))
            & (gap >= float(brake_tail_min_candidate_gap))
            & (gap <= float(brake_tail_max_candidate_gap))
            & (dev >= float(brake_tail_min_nominal_deviation))
            & tail_shape
            & bool(nominal_gate)
        )
        brake_tail_rescue_certified[ni] = False
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            brake_tail_rescue_certified &= intervention_macro_mask
        # Treat residual brake as a brake-family certificate for admission, but
        # retain the separate mask for diagnostics/reason strings.
        brake_rescue_certified = brake_rescue_certified | brake_tail_rescue_certified

    # v35: strict subset of residual-tail certificates allowed to bypass the
    # relative learned-PCD gain when challenging an already admitted nominal.
    # This is deliberately narrower than ``brake_tail_rescue_certified`` so the
    # selector does not reopen the v34 failure mode of frequent nominal-best
    # brake overrides.
    brake_tail_challenge_certified = brake_tail_rescue_certified.copy()
    if bool(brake_tail_challenge_bypass_pcd_gain):
        challenge_tail_shape = (
            (gap >= float(brake_tail_challenge_high_gap_min))
            | (r_dep <= float(brake_tail_challenge_low_r_dep_max))
        )
        brake_tail_challenge_certified &= (
            (drs_proxy >= float(brake_tail_challenge_min_pred_drs))
            & (pcd_proxy >= float(brake_tail_challenge_min_pred_pcd))
            & (gap >= float(brake_tail_challenge_min_candidate_gap))
            & (gap <= float(brake_tail_challenge_max_candidate_gap))
            & challenge_tail_shape
        )
    else:
        brake_tail_challenge_certified[:] = False

    # v24 BMRC: predicted post-contact deployability certificate.  This is used
    # as an intermediate channel between strict scalar LCB admission and the
    # macro-only v23 rescue.  The score is the same compact deployability proxy
    # used in the paper-facing PCD metric: DRS * sigmoid(R_dep) * exp(-gap).
    # Unlike the v23 certificate, this channel can be applied to near-contact
    # as well as contact, and it can require a bounded utility cost so the
    # selector does not repeatedly brake just because the macro family is safe.
    if bool(pcd_rescue_certificate) and 0 <= int(nominal_index) < n:
        ni = int(nominal_index)
        pcd_macro_allow = _split_name_set(pcd_rescue_macro_allowlist)
        pcd_macro_block = _split_name_set(pcd_rescue_macro_blocklist)
        pcd_macro_mask = np.ones((n,), dtype=bool)
        if pcd_macro_allow:
            pcd_macro_mask &= np.asarray([m in pcd_macro_allow for m in macro_names], dtype=bool)
        if pcd_macro_block:
            pcd_macro_mask &= ~np.asarray([m in pcd_macro_block for m in macro_names], dtype=bool)
        pcd_macro_mask &= np.asarray([bool(m) for m in macro_names], dtype=bool)

        nominal_gate = True
        if bool(pcd_rescue_require_nominal_low_headroom):
            # v27 DDC: the old BMRC low-headroom gate used a pure OR over
            # rec/gap/DRS.  With pcd_rescue_nominal_drs_max=1.01 this was
            # effectively always true, so a rescue certificate could trigger
            # even when nominal looked deployable.  Use a configurable
            # k-of-3 evidence gate while keeping k=1 as the backward-compatible
            # default.
            low_axes = 0
            if rec_lcb[ni] <= float(pcd_rescue_nominal_rec_lcb_max):
                low_axes += 1
            if gap[ni] >= float(pcd_rescue_nominal_gap_min):
                low_axes += 1
            if drs_proxy[ni] <= float(pcd_rescue_nominal_drs_max):
                low_axes += 1
            nominal_gate = low_axes >= max(1, int(pcd_rescue_nominal_low_headroom_min_axes or 1))
        if bool(pcd_rescue_require_nominal_unadmitted):
            nominal_gate = bool(nominal_gate) and (not bool(scalar_admitted[ni] or option_certified[ni] or relative_certified[ni] or protective_certified[ni] or brake_rescue_certified[ni]))

        utility_drop = utility[ni] - utility
        utility_ok = np.ones((n,), dtype=bool)
        if float(pcd_rescue_max_utility_drop) >= 0.0:
            utility_ok = (utility_drop <= float(pcd_rescue_max_utility_drop)) | (pcd_gain_vs_nom >= float(pcd_rescue_large_pcd_gain))

        pcd_rescue_certified = (
            feasible
            & (hard <= float(pcd_rescue_max_hard))
            & (harm <= float(pcd_rescue_max_harm))
            & pcd_macro_mask
            & (pcd_proxy >= float(pcd_rescue_min_pred_pcd))
            & (drs_proxy >= float(pcd_rescue_min_pred_drs))
            & (r_dep >= float(pcd_rescue_min_pred_r_dep))
            & (gap >= float(pcd_rescue_min_candidate_gap))
            & (gap <= float(pcd_rescue_max_candidate_gap))
            & utility_ok
            & bool(nominal_gate)
        )
        if float(pcd_rescue_min_pcd_gain) >= 0.0:
            pcd_rescue_certified &= pcd_gain_vs_nom >= float(pcd_rescue_min_pcd_gain)
        pcd_rescue_certified[ni] = False
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            pcd_rescue_certified &= intervention_macro_mask

    admitted = scalar_admitted | option_certified | relative_certified | protective_certified | brake_rescue_certified | pcd_rescue_certified
    if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
        admitted = admitted & intervention_macro_mask
        if 0 <= int(nominal_index) < n and (scalar_admitted[int(nominal_index)] or safe[int(nominal_index)]):
            admitted[int(nominal_index)] = True
    idxs = np.arange(n)
    intervention = (idxs != int(nominal_index)).astype(float)
    regime = (regime_name or "").lower()
    is_safe_regime = "safe" in regime or "normal" in regime or "background" in regime
    is_contact_regime = "contact" in regime
    try:
        cooldown_active = int(intervention_cooldown_steps or 0) > 0 and steps_since_last_intervention is not None and float(steps_since_last_intervention) < float(intervention_cooldown_steps)
    except Exception:
        cooldown_active = False

    # Continuous recovery shortfall; finite guard prevents NaNs from dominating.
    rec_shortfall = np.maximum(0.0, float(gamma_rec) - rec_lcb)
    rec_shortfall = np.where(np.isfinite(rec_shortfall), rec_shortfall, 1.0e6)
    score = (
        utility
        - float(shortfall_penalty) * rec_shortfall
        - float(gap_penalty) * gap
        - float(intervention_penalty) * intervention
        - float(deviation_penalty) * dev
        + float(recovery_bonus) * (rec_lcb - float(gamma_rec))
        + float(admission_bonus) * admitted.astype(float)
        + float(deployability_bonus) * drs_proxy
    )
    if bool(relative_recovery_certificate) and float(relative_recovery_bonus) != 0.0:
        rel_adv = np.maximum(0.0, rec_gain_vs_nom) + 0.25 * np.maximum(0.0, drs_gain_vs_nom) + 0.10 * np.maximum(0.0, gap_reduction_vs_nom)
        score = score + float(relative_recovery_bonus) * rel_adv * relative_certified.astype(float)
    if bool(protective_macro_certificate) and float(protective_macro_bonus) != 0.0:
        prot_adv = (
            float(protective_macro_score_rec_weight) * np.maximum(0.0, rec_gain_vs_nom)
            + float(protective_macro_score_drs_weight) * np.maximum(0.0, drs_gain_vs_nom)
            + float(protective_macro_score_gap_weight) * np.maximum(0.0, gap_reduction_vs_nom)
        )
        score = score + float(protective_macro_bonus) * prot_adv * protective_certified.astype(float)
    if bool(pcd_rescue_certificate) and float(pcd_rescue_bonus) != 0.0:
        score = score + float(pcd_rescue_bonus) * np.maximum(0.0, pcd_gain_vs_nom) * pcd_rescue_certified.astype(float)
    if is_contact_regime:
        score = score + float(contact_deployability_bonus) * drs_proxy - float(contact_gap_penalty) * gap

    if bool(relative_recovery_certificate) and float(relative_recovery_max_intervention_score_gain) >= 0.0 and 0 <= int(nominal_index) < n:
        # Cap the admission frontier, not the final ranking pool.  A candidate that
        # only looks attractive because of a large learned-score bonus should not
        # bypass the nominal-preserving abstention rule.
        relative_certified &= (score - score[int(nominal_index)]) <= float(relative_recovery_max_intervention_score_gain)
        admitted = scalar_admitted | option_certified | relative_certified | protective_certified | brake_rescue_certified | pcd_rescue_certified
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            admitted = admitted & intervention_macro_mask
            if 0 <= int(nominal_index) < n and (scalar_admitted[int(nominal_index)] or safe[int(nominal_index)]):
                admitted[int(nominal_index)] = True

    # v42 kept the value head preference-only, which made it unusable whenever
    # the independent admission set contained nominal alone. v43 OC-RSC may
    # promote exactly the deterministic top-1 actionable candidate when its
    # score crosses a separately fitted and held-out-verified risk threshold.
    # Feasibility, hard/harm, macro and trajectory-actionability gates have
    # already been applied above. Safe buckets keep this switch disabled.
    if bool(direct_value_risk_controlled_admission):
        admitted = admitted | direct_value_challenge
    else:
        direct_value_challenge &= admitted
    if bool(direct_value_certificate) and float(direct_value_bonus) != 0.0:
        # v44: reward certificate margin above its fitted regime-specific
        # threshold, not raw advantage above zero. A valid selective rule may
        # legitimately use a negative score threshold, and v43's max(0, raw)
        # bonus could then admit a verified candidate without ever letting it
        # beat nominal. The unit offset makes certificate use observable while
        # preserving all physical, opportunity and held-out risk gates.
        cert_margin = np.maximum(0.0, direct_advantage_lcb - float(direct_value_min_advantage_lcb))
        score = score + float(direct_value_bonus) * (1.0 + cert_margin) * direct_value_challenge.astype(float)
        if bool(direct_value_risk_controlled_admission) and 0 <= int(nominal_index) < n and bool(direct_value_challenge.any()):
            # Reproduce the calibrated deployment event exactly: a verified
            # deterministic top-1 challenge must outrank nominal. It can still
            # lose to a stronger independently certified recovery candidate.
            ni = int(nominal_index)
            floor = float(score[ni]) + max(1.0e-4, float(switch_score_margin) + 1.0e-4)
            score[direct_value_challenge] = np.maximum(score[direct_value_challenge], floor)

    # If the current rollout already exceeded the intervention budget, increase
    # the marginal cost of deviating from nominal.  This is intentionally a soft
    # budget because emergency recovery should still be selectable.
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            used = float(intervention_budget_used or 0.0)
            steps = max(1.0, float(intervention_budget_steps or 1.0))
            target = float(intervention_budget_rate)
            over = max(0.0, (used / steps) - target)
            if over > 0.0:
                score = score - float(intervention_budget_penalty) * over * intervention
        except Exception:
            pass

    if safe.any():
        pool = safe.copy()
    elif feasible.any():
        # Preserve the original hard/harm safety priority when no fully safe
        # candidate exists.
        cand = np.where(feasible)[0]
        min_hard = float(np.min(hard[cand]))
        near_hard = feasible & (hard <= min_hard + 1.0e-6)
        cand2 = np.where(near_hard)[0]
        min_harm = float(np.min(harm[cand2])) if cand2.size else float(np.min(harm[cand]))
        pool = near_hard & (harm <= min_harm + 1.0e-6)
    else:
        pool = np.ones((n,), dtype=bool)

    # Safe/NUP gate.  In benign regimes the paper claim is nominal utility
    # preservation, not recovery maximization.  The learned recovery heads can be
    # pessimistic on safe_v2 because the stress regimes dominate negative
    # deployability.  Therefore, when the nominal prefix is dynamically feasible
    # and the active regime is safe/background, allow an explicit hard lock that
    # bypasses noisy recovery-head shortfall.
    if bool(safe_force_nominal_when_feasible) and is_safe_regime and 0 <= nominal_index < n:
        # In safe/background regimes, nominal preservation should be a certifiable
        # behavior, not just an unconditional heuristic.  `always` keeps the old
        # hard lock for ablation/papercheck; `certified` locks nominal only when
        # the learned heads and feasibility metadata agree that nominal is already
        # deployable and low-gap.
        force_mode = str(safe_force_nominal_mode or "feasible").strip().lower()
        certified = (
            pool[nominal_index]
            and drs_proxy[nominal_index] >= float(safe_cert_min_pred_drs)
            and gap[nominal_index] <= float(safe_cert_max_pred_gap)
            and rec_lcb[nominal_index] >= float(gamma_rec) - float(safe_cert_rec_slack)
        )
        allow_force = (
            force_mode in {"always", "present", "nominal", "hard"}
            or (pool[nominal_index] and force_mode in {"", "feasible", "safe"})
            or (certified and force_mode in {"cert", "certified", "learned", "model", "guarded"})
        )
        if allow_force:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            reason = "nominal_safe_certified_locked" if force_mode in {"cert", "certified", "learned", "model", "guarded"} else "nominal_safe_force_locked"
            return SelectionResult(int(nominal_index), reason, admitted)

    # Stress-regime option: when calibrated-admitted candidates exist, rank only
    # within them.  The default remains soft-constrained for backward
    # compatibility, but near-contact/contact experiments should set
    # prefer_admitted_by_bucket=true so DRS is not sacrificed to high utility.
    rank_pool = pool & admitted if bool(prefer_admitted) and bool((pool & admitted).any()) else pool

    cand = np.where(rank_pool)[0]
    if cand.size == 0:
        cand = np.arange(n)
    best_idx = int(cand[np.argmax(score[cand])])

    # Admitted-intervention abstention.  OC-RAP's claim is deployable recovery,
    # so executing a recovery prefix that the calibrated selector itself did not
    # admit is hard to defend in paper experiments.  When enabled, intervention
    # is allowed only if a candidate passes calibrated admission and optional
    # predicted shared-action DRS/gap guards.  If none is certified, preserve
    # nominal rather than taking a high-utility but uncertified recovery action.
    if bool(require_admitted_intervention):
        certified_intervention = admitted.copy()
        if intervention_macro_allow or intervention_macro_block or bool(intervention_require_macro):
            certified_intervention &= intervention_macro_mask
        if float(intervention_min_pred_drs) >= 0.0:
            certified_intervention &= drs_proxy >= float(intervention_min_pred_drs)
        if float(intervention_max_pred_gap) >= 0.0:
            certified_intervention &= gap <= float(intervention_max_pred_gap)
        # For relative recovery in post-contact/near-contact regimes, a certified
        # recovery maneuver may lie outside the nominal hard-rule pool (e.g. a
        # stabilization/pull-over action).  Do not discard such a candidate after
        # certifying it via the explicit recovery pool above.
        certified_intervention &= (pool | relative_certified | protective_certified | brake_rescue_certified | pcd_rescue_certified)
        certified_non_nom = certified_intervention.copy()
        if 0 <= nominal_index < n:
            certified_non_nom[int(nominal_index)] = False

        # v14: certified intervention must not merely pass an absolute
        # threshold; it should have some predicted evidence that switching away
        # from nominal improves deployable recoverability or closes the
        # oracle--deployability gap.  This keeps OC-RAP from reducing FRA/ODG by
        # taking low-DRS recovery prefixes and turns abstention into an explicit
        # learned/certified decision.
        if bool(require_intervention_evidence) and 0 <= nominal_index < n:
            rec_gain = rec_lcb - rec_lcb[int(nominal_index)]
            drs_gain = drs_proxy - drs_proxy[int(nominal_index)]
            gap_reduction = gap[int(nominal_index)] - gap
            evidence = (
                (rec_gain >= float(intervention_min_rec_lcb_gain))
                | (drs_gain >= float(intervention_min_drs_gain))
                | (gap_reduction >= float(intervention_min_gap_reduction))
            )
            if bool(option_drs_certificate_counts_as_evidence):
                evidence = evidence | option_certified
            if bool(relative_recovery_counts_as_evidence):
                evidence = evidence | relative_certified
            if bool(protective_macro_counts_as_evidence):
                evidence = evidence | protective_certified
            if bool(brake_rescue_counts_as_evidence):
                evidence = evidence | brake_rescue_certified
            if bool(pcd_rescue_counts_as_evidence):
                evidence = evidence | pcd_rescue_certified
            if bool(direct_value_counts_as_evidence):
                evidence = evidence | direct_value_challenge
            certified_non_nom &= evidence

        if bool(intervention_budget_hard) and intervention_budget_rate is not None and intervention_budget_steps not in {None, 0} and 0 <= nominal_index < n:
            try:
                used = float(intervention_budget_used or 0.0)
                steps = max(1.0, float(intervention_budget_steps or 1.0))
                budget_exceeded_now = (used / steps) >= float(intervention_budget_rate)
            except Exception:
                budget_exceeded_now = False
            if budget_exceeded_now:
                rec_gain = rec_lcb - rec_lcb[int(nominal_index)]
                drs_gain = drs_proxy - drs_proxy[int(nominal_index)]
                gap_reduction = gap[int(nominal_index)] - gap
                hard_budget_evidence = (
                    (rec_gain >= float(intervention_budget_hard_min_rec_gain))
                    | (drs_gain >= float(intervention_budget_hard_min_drs_gain))
                    | (gap_reduction >= float(intervention_budget_hard_min_gap_reduction))
                    | protective_certified
                    | relative_certified
                )
                if bool(brake_rescue_budget_bypass):
                    hard_budget_evidence = hard_budget_evidence | brake_rescue_certified
                if bool(pcd_rescue_budget_bypass):
                    hard_budget_evidence = hard_budget_evidence | pcd_rescue_certified
                # v36: only the strict residual-tail challenge certificate may
                # override a consumed exposure budget.  The broad brake rescue
                # certificate is intentionally not enough because it caused the
                # v34/v35 offline high-frequency brake policy.
                if bool(brake_tail_challenge_budget_bypass):
                    hard_budget_evidence = hard_budget_evidence | brake_tail_challenge_certified
                if bool(direct_value_counts_as_evidence):
                    hard_budget_evidence = hard_budget_evidence | direct_value_challenge
                certified_non_nom &= hard_budget_evidence

        # v25 closed-loop exposure gate.  A rescue certificate may override a
        # single over-confident nominal decision, but it should not create a
        # burst of brake prefixes at consecutive replanning steps.  The gate is
        # applied after semantic/evidence certification so nominal is preserved
        # when the exposure budget says another intervention is premature.
        if bool(cooldown_active):
            if bool(brake_tail_challenge_cooldown_bypass):
                certified_non_nom &= brake_tail_challenge_certified
            else:
                certified_non_nom[:] = False

        if bool(certified_non_nom.any()):
            cc = np.where(certified_non_nom)[0]
            best_idx = int(cc[np.argmax(score[cc])])
        elif 0 <= nominal_index < n and bool(unadmitted_fallback_to_nominal):
            # Important: do not require nominal to be in the temporary hard/harm
            # pool.  In closed-loop feature-only runs the nominal metadata can
            # be noisy; executing an uncertified recovery is harder to defend
            # than abstaining to nominal.  This fixes the v13 failure where
            # require_admitted_intervention=true still produced
            # best_calibrated_soft_constraint interventions.
            admitted = admitted.copy()
            admitted[int(nominal_index)] = True
            return SelectionResult(int(nominal_index), "nominal_no_certified_intervention_preserved", admitted)
        elif not admitted[best_idx]:
            admitted_pool = np.where(pool & admitted)[0]
            if admitted_pool.size > 0:
                best_idx = int(admitted_pool[np.argmax(score[admitted_pool])])
            elif 0 <= nominal_index < n:
                admitted = admitted.copy()
                admitted[int(nominal_index)] = True
                return SelectionResult(int(nominal_index), "nominal_no_certified_intervention_preserved", admitted)

    # Stress DRS guard.  A recurring failure mode in the current results is that
    # learned OC-RAP lowers FRA/ODG but sacrifices executed DRS.  When configured,
    # do not switch away from a feasible nominal prefix to a candidate whose
    # predicted shared-option success is materially worse than nominal.  Keep the
    # default disabled because early recovery may deliberately trade nominal
    # behavior for safety.
    if (not is_safe_regime) and float(stress_preserve_nominal_min_drs_drop) >= 0.0 and 0 <= nominal_index < n and pool[nominal_index]:
        if drs_proxy[best_idx] + float(stress_preserve_nominal_min_drs_drop) < drs_proxy[nominal_index]:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_stress_drs_guard", admitted)

    # Stress-regime nominal anchor.  Near-contact scenes are still benign until
    # impact, and contact scenes often have a high-quality nominal recovery
    # baseline.  Avoid switching away from a predicted-good nominal prefix unless
    # the learned candidate brings a material deployability/gap advantage.  This
    # is intentionally optional and should be reported as a cautious-selector
    # ablation, not hidden as a universal rule.
    if bool(stress_nominal_anchor) and (not is_safe_regime) and 0 <= nominal_index < n and pool[nominal_index]:
        nominal_anchor_ok = (
            drs_proxy[nominal_index] >= float(stress_anchor_drs_floor)
            and gap[nominal_index] <= float(stress_anchor_max_gap)
            and rec_lcb[nominal_index] >= float(gamma_rec) - float(stress_anchor_rec_slack)
        )
        if nominal_anchor_ok and best_idx != int(nominal_index):
            drs_gain = float(drs_proxy[best_idx] - drs_proxy[nominal_index])
            rec_gain = float(rec_lcb[best_idx] - rec_lcb[nominal_index])
            gap_reduction = float(gap[nominal_index] - gap[best_idx])
            if (
                drs_gain < float(stress_anchor_min_drs_gain)
                and rec_gain < float(stress_anchor_min_rec_gain)
                and gap_reduction < float(stress_anchor_min_gap_reduction)
                and not bool(direct_value_challenge[best_idx])
            ):
                admitted = admitted.copy()
                admitted[nominal_index] = True
                return SelectionResult(int(nominal_index), "nominal_stress_anchor_preserved", admitted)

    # Nominal-preserving calibration.  In safe/background regimes, recovery
    # shortfall by itself should not trigger large behavior changes unless the
    # alternative is clearly better in calibrated score and materially improves
    # deployable-recovery LCB or oracle-deployable gap.
    nominal_extra_slack = float(safe_nominal_slack if is_safe_regime else nominal_slack)
    budget_exceeded = False
    if intervention_budget_rate is not None and intervention_budget_steps not in {None, 0}:
        try:
            budget_exceeded = (float(intervention_budget_used or 0.0) / max(1.0, float(intervention_budget_steps or 1.0))) >= float(intervention_budget_rate)
        except Exception:
            budget_exceeded = False

    if 0 <= nominal_index < n and pool[nominal_index]:
        nom_gap_ok = gap[nominal_index] <= float(nominal_slack_gap_limit)
        nom_near_gamma = rec_lcb[nominal_index] >= float(gamma_rec) - nominal_extra_slack
        nom_near_score = score[nominal_index] >= score[best_idx] - float(nominal_utility_slack)
        if admitted[nominal_index]:
            if bool(stress_rescue_challenge_nominal) and (not is_safe_regime):
                challenge_mask = (brake_rescue_certified | pcd_rescue_certified | (direct_value_challenge & bool(direct_value_challenge_nominal))) & admitted
                challenge_mask[int(nominal_index)] = False
                # Apply the v26 guarded challenge frontier.  This is deliberately
                # separate from certification: a rescue can still be admitted when
                # nominal is unadmitted, but challenging an already-admitted nominal
                # requires stronger evidence and bounded exposure.
                if float(rescue_challenge_min_candidate_gap) >= 0.0:
                    challenge_mask &= (gap >= float(rescue_challenge_min_candidate_gap)) | direct_value_challenge
                if float(rescue_challenge_max_candidate_gap) >= 0.0:
                    challenge_mask &= (gap <= float(rescue_challenge_max_candidate_gap)) | direct_value_challenge
                if float(rescue_challenge_min_pred_drs) >= 0.0:
                    challenge_mask &= (drs_proxy >= float(rescue_challenge_min_pred_drs)) | direct_value_challenge
                if float(rescue_challenge_min_pred_pcd) >= 0.0:
                    challenge_mask &= (pcd_proxy >= float(rescue_challenge_min_pred_pcd)) | direct_value_challenge
                # v27 DDC: challenge is a stronger act than certification.
                # A candidate can be rescue-certified when nominal is not
                # admitted, but replacing an already admitted nominal prefix
                # requires regime-conditioned dominance in the same coordinates
                # used by the paper metrics: PCD, deployable margin, DRS, and
                # oracle-to-deployable gap.
                ch_allow = _split_name_set(rescue_challenge_macro_allowlist)
                ch_block = _split_name_set(rescue_challenge_macro_blocklist)
                if ch_allow:
                    challenge_mask &= np.asarray([m in ch_allow for m in macro_names], dtype=bool)
                if ch_block:
                    challenge_mask &= ~np.asarray([m in ch_block for m in macro_names], dtype=bool)
                axis_count = np.zeros((n,), dtype=int)
                if float(rescue_challenge_min_pcd_gain) >= 0.0:
                    ok = pcd_gain_vs_nom >= float(rescue_challenge_min_pcd_gain)
                    # Residual brake-tail is specifically designed for cases where
                    # relative learned PCD is inverted by nominal over-confidence.
                    # Keep the absolute tail certificate gates, but do not require
                    # positive relative PCD gain for these already-certified brake
                    # candidates when explicitly enabled.
                    if bool(brake_tail_challenge_bypass_pcd_gain):
                        ok = ok | brake_tail_challenge_certified | direct_value_challenge
                    challenge_mask &= ok
                    axis_count += ok.astype(int)
                if float(rescue_challenge_min_rec_lcb_gain) >= 0.0:
                    ok = (rec_gain_vs_nom >= float(rescue_challenge_min_rec_lcb_gain)) | direct_value_challenge
                    challenge_mask &= ok if int(rescue_challenge_min_improvement_axes or 0) <= 0 else challenge_mask
                    axis_count += ok.astype(int)
                if float(rescue_challenge_min_drs_gain) >= 0.0:
                    ok = (drs_gain_vs_nom >= float(rescue_challenge_min_drs_gain)) | direct_value_challenge
                    challenge_mask &= ok if int(rescue_challenge_min_improvement_axes or 0) <= 0 else challenge_mask
                    axis_count += ok.astype(int)
                if float(rescue_challenge_min_gap_reduction) >= 0.0:
                    ok = (gap_reduction_vs_nom >= float(rescue_challenge_min_gap_reduction)) | direct_value_challenge
                    challenge_mask &= ok if int(rescue_challenge_min_improvement_axes or 0) <= 0 else challenge_mask
                    axis_count += ok.astype(int)
                if int(rescue_challenge_min_improvement_axes or 0) > 0:
                    challenge_mask &= (axis_count >= int(rescue_challenge_min_improvement_axes)) | direct_value_challenge
                if float(rescue_challenge_max_pred_utility) >= 0.0:
                    challenge_mask &= utility <= float(rescue_challenge_max_pred_utility)
                if int(rescue_challenge_max_used) >= 0:
                    try:
                        challenge_mask &= float(intervention_budget_used or 0.0) < float(rescue_challenge_max_used)
                    except Exception:
                        pass
                if int(brake_tail_challenge_max_consecutive) >= 0:
                    prev_macro = str(previous_selected_macro or "").strip().lower()
                    try:
                        prev_run = int(same_macro_run_length or 0)
                    except Exception:
                        prev_run = 0
                    if prev_macro == str(brake_rescue_macro_name).strip().lower() and prev_run >= int(brake_tail_challenge_max_consecutive):
                        # Cooldown bypass in v38 can otherwise produce a long run
                        # of fresh brake selections. Cap only the strict tail
                        # challenge; ordinary admitted/recovery logic remains.
                        challenge_mask &= ~brake_tail_challenge_certified
                if bool(cooldown_active):
                    if bool(brake_tail_challenge_cooldown_bypass):
                        challenge_mask &= brake_tail_challenge_certified
                    else:
                        challenge_mask[:] = False
                # Optional nominal guard: if nominal already has a very high
                # learned PCD proxy and low predicted gap, do not challenge it.
                # This prevents v25's failure mode where a brake with lower
                # predicted deployability replaced a good nominal prefix.
                nom_pcd = float(pcd_proxy[int(nominal_index)])
                nom_gap = float(gap[int(nominal_index)])
                if (
                    float(rescue_challenge_nominal_guard_min_pcd) >= 0.0
                    and float(rescue_challenge_nominal_guard_max_gap) >= 0.0
                    and nom_pcd >= float(rescue_challenge_nominal_guard_min_pcd)
                    and nom_gap <= float(rescue_challenge_nominal_guard_max_gap)
                ):
                    # Let only candidates with a material learned PCD advantage
                    # break this strong nominal certificate.
                    min_gain = max(float(rescue_challenge_min_pcd_gain), 0.035)
                    challenge_mask &= (pcd_gain_vs_nom >= min_gain) | direct_value_challenge
                if bool(challenge_mask.any()):
                    cc = np.where(challenge_mask)[0]
                    challenge_score = (
                        float(rescue_challenge_score_pcd_weight) * pcd_proxy
                        + float(direct_value_bonus) * np.maximum(0.0, direct_advantage_lcb)
                        + float(rescue_challenge_score_drs_weight) * drs_proxy
                        - float(rescue_challenge_score_gap_weight) * gap
                        - float(rescue_challenge_score_utility_weight) * np.maximum(0.0, utility - utility[int(nominal_index)])
                    )
                    chosen = int(cc[np.argmax(challenge_score[cc])])
                    if direct_value_challenge[chosen]:
                        return SelectionResult(chosen, "best_direct_value_lcb_guarded_challenge", admitted)
                    if pcd_rescue_certified[chosen] and not brake_rescue_certified[chosen]:
                        return SelectionResult(chosen, "best_pcd_rescue_guarded_challenge", admitted)
                    if brake_tail_rescue_certified[chosen] and not (brake_rescue_certified[chosen] and not brake_tail_rescue_certified[chosen]):
                        return SelectionResult(chosen, "best_brake_tail_rescue_guarded_challenge", admitted)
                    return SelectionResult(chosen, "best_brake_rescue_guarded_challenge", admitted)
            return SelectionResult(int(nominal_index), "nominal_calibrated_admitted", admitted)
        if bool(budget_preserve_nominal) and budget_exceeded and nom_gap_ok and rec_lcb[nominal_index] >= float(gamma_rec) - float(budget_nominal_slack):
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_budget_preserved", admitted)
        if is_safe_regime and nom_gap_ok and nom_near_gamma:
            score_gain = float(score[best_idx] - score[nominal_index])
            rec_gain = float(rec_lcb[best_idx] - rec_lcb[nominal_index])
            gap_reduction = float(gap[nominal_index] - gap[best_idx])
            drs_gain = float(drs_proxy[best_idx] - drs_proxy[nominal_index])
            material_recovery_gain = rec_gain >= float(safe_min_rec_lcb_gain)
            material_gap_gain = gap_reduction >= float(safe_min_gap_reduction)
            material_drs_gain = drs_gain >= float(safe_min_drs_gain)
            if bool(safe_hard_nominal_guard) and gap[nominal_index] <= float(safe_nominal_max_gap):
                # Safe_v2 is the nominal-preservation regime.  Do not switch for
                # a single noisy headroom advantage; require a score improvement
                # plus consistent recovery/gap evidence.
                enough_evidence = (material_recovery_gain and material_gap_gain) if bool(safe_override_require_both) else (material_recovery_gain or material_gap_gain or material_drs_gain)
                if (not enough_evidence) or score_gain < float(safe_switch_score_margin):
                    admitted = admitted.copy()
                    admitted[nominal_index] = True
                    return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
            # Safe/background is a nominal-preserving operating region.  Switch
            # away from nominal only when the alternative is not just higher
            # scoring, but meaningfully improves calibrated recoverability or
            # reduces oracle-deployable gap.
            if (not material_recovery_gain and not material_gap_gain and not material_drs_gain) or score_gain < float(safe_switch_score_margin):
                admitted = admitted.copy()
                admitted[nominal_index] = True
                return SelectionResult(int(nominal_index), "nominal_safe_switch_guard", admitted)
        if nom_gap_ok and nom_near_gamma and nom_near_score:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_calibrated_preserved", admitted)
        if (not is_safe_regime) and float(score[best_idx] - score[nominal_index]) < float(switch_score_margin) and nom_gap_ok and nom_near_gamma:
            admitted = admitted.copy()
            admitted[nominal_index] = True
            return SelectionResult(int(nominal_index), "nominal_switch_margin_preserved", admitted)

    reason = "best_calibrated_admitted_score" if admitted[best_idx] else "best_calibrated_soft_constraint"
    if admitted[best_idx] and direct_value_challenge[best_idx]:
        reason = "best_direct_value_lcb_score"
    elif admitted[best_idx] and pcd_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx] and not brake_rescue_certified[best_idx]:
        reason = "best_pcd_rescue_certified_score"
    elif admitted[best_idx] and brake_tail_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
        reason = "best_brake_tail_rescue_certified_score"
    elif admitted[best_idx] and brake_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
        reason = "best_brake_rescue_certified_score"
    elif admitted[best_idx] and protective_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx]:
        reason = "best_protective_macro_recovery_certified_score"
    elif admitted[best_idx] and relative_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx]:
        reason = "best_relative_recovery_certified_score"
    elif admitted[best_idx] and option_certified[best_idx] and not scalar_admitted[best_idx]:
        reason = "best_option_drs_certified_score"
    admitted_rank_pool = pool | relative_certified | protective_certified | brake_rescue_certified | pcd_rescue_certified | direct_value_challenge
    if bool(prefer_admitted) and bool((admitted_rank_pool & admitted).any()) and admitted[best_idx]:
        if direct_value_challenge[best_idx]:
            reason = "best_direct_value_lcb_prefer_admitted"
        elif pcd_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx] and not brake_rescue_certified[best_idx]:
            reason = "best_pcd_rescue_certified_prefer_admitted"
        elif brake_tail_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
            reason = "best_brake_tail_rescue_certified_prefer_admitted"
        elif brake_rescue_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx] and not protective_certified[best_idx]:
            reason = "best_brake_rescue_certified_prefer_admitted"
        elif protective_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx] and not relative_certified[best_idx]:
            reason = "best_protective_macro_recovery_certified_prefer_admitted"
        elif relative_certified[best_idx] and not scalar_admitted[best_idx] and not option_certified[best_idx]:
            reason = "best_relative_recovery_certified_prefer_admitted"
        elif option_certified[best_idx] and not scalar_admitted[best_idx]:
            reason = "best_option_drs_certified_prefer_admitted"
        else:
            reason = "best_calibrated_prefer_admitted"
    return SelectionResult(best_idx, reason, admitted)

