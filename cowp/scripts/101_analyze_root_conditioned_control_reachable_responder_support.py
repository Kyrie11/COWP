from __future__ import annotations

"""Analyze V16.8.44 root-conditioned control-reachable responder support under the frozen development protocol.

The six-item outcome gate is inherited verbatim from V16.8.39--41.  Root support,
response-envelope feasibility, and interaction-only selections are mechanism
attribution diagnostics; they are deliberately not added to the promotion gate.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHOD_KEY = "root_conditioned_control_reachable_responder_support"
CANONICAL_METHOD = "cowp_root_conditioned_control_reachable_responder_support"


def _load(path: str | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text()) if path else None


def _rows(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_id"]): row for row in data.get("scenario_results", [])}


def _metric(row: dict[str, Any], key: str) -> float:
    value = (row.get("standard_metrics", {}) or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("nan")


def _event(row: dict[str, Any], key: str) -> bool:
    value = _metric(row, key)
    return bool(math.isfinite(value) and value > 0.0)


def _diag_value(row: dict[str, Any], key: str) -> float | None:
    value = (row.get("diagnostics", {}) or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mcnemar_exact(rescued: int, induced: int) -> float:
    n = int(rescued + induced)
    if n <= 0:
        return 1.0
    k = min(int(rescued), int(induced))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int = 16844,
    draws: int = 10_000,
) -> list[float] | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        means[i] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


_DIAGNOSTIC_KEYS = [
    "fallback_step_rate",
    "zero_conventional_candidate_step_rate",
    "mean_conventional_candidates",
    "mean_valid_candidates",
    "mean_max_collision_safe_prefix_steps",
    "mean_selected_collision_safe_prefix_steps",
    "recovery_switch_step_rate",
    "selected_waymax_kinematic_feasible_step_rate",
    "selected_waymax_kinematic_feasible_rate_on_recovery_switch_steps",
    "interaction_aware_reachable_response_envelope_step_rate",
    "recovery_tube_probe_step_rate",
    "recovery_tube_certificate_step_rate",
    "recovery_tube_action_change_step_rate",
    "recovery_tube_lifted_selection_rate_on_certified_steps",
    "recovery_tube_event_release_selection_rate_on_certified_steps",
    "recovery_tube_nested_v39_selection_rate_on_certified_steps",
    "recovery_tube_interaction_attempt_step_rate",
    "recovery_tube_interaction_selection_rate_on_certified_steps",
    "mean_recovery_tube_interaction_support_agents_total_on_attempts",
    "mean_recovery_tube_interaction_support_agents_ready_on_attempts",
    "mean_recovery_tube_interaction_retained_roots_on_attempts",
    "mean_recovery_tube_interaction_eligible_profiles_on_attempts",
    "mean_recovery_tube_interaction_hypotheses_on_attempts",
    "mean_recovery_tube_interaction_noop_hypotheses_on_attempts",
    "mean_recovery_tube_interaction_no_blocker_rejects_on_attempts",
    "mean_recovery_tube_interaction_unsupported_blocker_rejects_on_attempts",
    "mean_recovery_tube_interaction_residual_physical_rejects_on_attempts",
    "mean_recovery_tube_interaction_root_unrecoverable_rejects_on_attempts",
    "mean_recovery_tube_interaction_joint_incompatibility_rejects_on_attempts",
    "mean_recovery_tube_interaction_environment_compatibility_checks_on_attempts",
    "mean_recovery_tube_interaction_environment_compatibility_rejects_on_attempts",
    "mean_recovery_tube_interaction_joint_compatibility_checks_on_attempts",
    "mean_recovery_tube_interaction_joint_compatibility_rejects_on_attempts",
    "mean_recovery_tube_interaction_joint_assignment_backtracks_on_attempts",
    "mean_recovery_tube_interaction_selected_blockers_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_roots_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_minimum_root_mass_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_maximum_response_burden_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_profile_evaluations_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_environment_agents_on_interaction_steps",
    "mean_recovery_tube_interaction_selected_environment_checks_on_interaction_steps",
    "recovery_tube_nested_v42_selection_rate_on_certified_steps",
    "recovery_tube_blocker_query_attempt_step_rate",
    "recovery_tube_blocker_query_selection_rate_on_certified_steps",
    "mean_recovery_tube_blocker_query_agents_on_attempts",
    "mean_recovery_tube_blocker_query_ready_agents_on_attempts",
    "mean_recovery_tube_blocker_query_hypotheses_on_attempts",
    "mean_recovery_tube_blocker_query_unsupported_rejects_on_attempts",
    "mean_recovery_tube_blocker_query_root_unrecoverable_rejects_on_attempts",
    "mean_recovery_tube_blocker_query_control_reachable_response_attempts_on_attempts",
    "mean_recovery_tube_blocker_query_control_reachable_response_profiles_found_on_attempts",
    "mean_recovery_tube_blocker_query_control_reachable_response_profile_evaluations_on_attempts",
    "mean_recovery_tube_blocker_query_control_reachable_response_selected_roots_on_attempts",
    "mean_recovery_tube_blocker_query_environment_cache_hits_on_attempts",
    "mean_recovery_tube_blocker_query_joint_cache_hits_on_attempts",
    "mean_recovery_tube_blocker_query_successor_context_cache_hits_on_attempts",
    "mean_recovery_tube_parent_pool_on_probes",
    "mean_recovery_tube_parent_action_classes_on_probes",
    "mean_recovery_tube_parents_with_nominal_conflict_on_probes",
    "mean_recovery_tube_hypotheses_generated_on_probes",
    "mean_recovery_tube_unique_action_hypotheses_on_probes",
    "mean_recovery_tube_full_physically_safe_on_probes",
    "mean_recovery_tube_shift_closed_on_probes",
    "mean_recovery_tube_abs_first_accel_delta_on_certified_steps",
    "mean_recovery_tube_selected_collision_margin_on_certified_steps",
    "mean_recovery_tube_selected_shift_collision_margin_on_certified_steps",
    "mean_recovery_tube_selected_fallback_score_delta_on_action_changes",
]


def _aggregate(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(ids)}
    for label, key in {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }.items():
        out[label] = float(np.mean([_event(rows[sid], key) for sid in ids])) if ids else 0.0
    ep = np.asarray([_metric(rows[sid], "EP") for sid in ids], dtype=np.float64)
    ep = ep[np.isfinite(ep)]
    out["EP"] = float(ep.mean()) if ep.size else None

    macro: dict[str, float | None] = {}
    for key in _DIAGNOSTIC_KEYS:
        values = [_diag_value(rows[sid], key) for sid in ids]
        finite = [value for value in values if value is not None]
        macro[key] = float(np.mean(finite)) if finite else None
    out["scenario_macro_average_diagnostics"] = macro
    return out


def _paired(
    base: dict[str, dict[str, Any]],
    alt: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, key in {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }.items():
        rescued = [sid for sid in ids if _event(base[sid], key) and not _event(alt[sid], key)]
        induced = [sid for sid in ids if not _event(base[sid], key) and _event(alt[sid], key)]
        shared = [sid for sid in ids if _event(base[sid], key) and _event(alt[sid], key)]
        out[label] = {
            "rescued": len(rescued),
            "induced": len(induced),
            "shared_failure": len(shared),
            "net_failures_removed": len(rescued) - len(induced),
            "mcnemar_exact_p": _mcnemar_exact(len(rescued), len(induced)),
            "rescued_ids": rescued,
            "induced_ids": induced,
        }

    deltas: list[float] = []
    per_scene: dict[str, float] = {}
    for sid in ids:
        before, after = _metric(base[sid], "EP"), _metric(alt[sid], "EP")
        if math.isfinite(before) and math.isfinite(after):
            delta = float(after - before)
            deltas.append(delta)
            per_scene[sid] = delta
    arr = np.asarray(deltas, dtype=np.float64)
    out["EP"] = {
        "paired_n": int(arr.size),
        "delta_mean": float(arr.mean()) if arr.size else None,
        "delta_median": float(np.median(arr)) if arr.size else None,
        "bootstrap95": _bootstrap_mean_ci(arr),
        "per_scene_delta": per_scene,
    }
    return out


def _count_from_rate(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
    rate_key: str,
    denominators: dict[str, int],
) -> int:
    total = 0
    for sid in ids:
        den = int(denominators.get(sid, 0))
        if den <= 0:
            continue
        value = _diag_value(rows[sid], rate_key)
        if value is not None:
            total += int(round(float(value) * den))
    return int(total)


def _weighted_total(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
    mean_key: str,
    denominators: dict[str, int],
) -> tuple[float, int]:
    total = 0.0
    den_total = 0
    for sid in ids:
        den = int(denominators.get(sid, 0))
        if den <= 0:
            continue
        value = _diag_value(rows[sid], mean_key)
        if value is None:
            continue
        total += float(value) * den
        den_total += den
    return float(total), int(den_total)


def _pooled_interaction_mechanism(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    steps: dict[str, int] = {}
    probes: dict[str, int] = {}
    certificates: dict[str, int] = {}
    changes: dict[str, int] = {}
    attempts: dict[str, int] = {}
    for sid in ids:
        diag = rows[sid].get("diagnostics", {}) or {}
        n = int(diag.get("steps", 0) or 0)
        steps[sid] = n
        probes[sid] = int(round(float(diag.get("recovery_tube_probe_step_rate", 0.0) or 0.0) * n))
        certificates[sid] = int(round(float(diag.get("recovery_tube_certificate_step_rate", 0.0) or 0.0) * n))
        changes[sid] = int(round(float(diag.get("recovery_tube_action_change_step_rate", 0.0) or 0.0) * n))
        attempts[sid] = int(round(float(diag.get("recovery_tube_interaction_attempt_step_rate", 0.0) or 0.0) * n))

    total_steps = int(sum(steps.values()))
    total_probes = int(sum(probes.values()))
    total_certificates = int(sum(certificates.values()))
    total_changes = int(sum(changes.values()))
    total_attempts = int(sum(attempts.values()))

    nested_selected = _count_from_rate(
        rows, ids, "recovery_tube_nested_v39_selection_rate_on_certified_steps", certificates
    )
    interaction_selected = _count_from_rate(
        rows, ids, "recovery_tube_interaction_selection_rate_on_certified_steps", certificates
    )
    lifted_selected = _count_from_rate(
        rows, ids, "recovery_tube_lifted_selection_rate_on_certified_steps", certificates
    )
    event_selected = _count_from_rate(
        rows, ids, "recovery_tube_event_release_selection_rate_on_certified_steps", certificates
    )

    attempt_keys = {
        "support_agents_total": "mean_recovery_tube_interaction_support_agents_total_on_attempts",
        "support_agents_ready": "mean_recovery_tube_interaction_support_agents_ready_on_attempts",
        "retained_roots": "mean_recovery_tube_interaction_retained_roots_on_attempts",
        "eligible_profiles": "mean_recovery_tube_interaction_eligible_profiles_on_attempts",
        "hypotheses_evaluated": "mean_recovery_tube_interaction_hypotheses_on_attempts",
        "noop_hypotheses_skipped": "mean_recovery_tube_interaction_noop_hypotheses_on_attempts",
        "no_blocker_rejects": "mean_recovery_tube_interaction_no_blocker_rejects_on_attempts",
        "unsupported_blocker_rejects": "mean_recovery_tube_interaction_unsupported_blocker_rejects_on_attempts",
        "residual_physical_rejects": "mean_recovery_tube_interaction_residual_physical_rejects_on_attempts",
        "root_unrecoverable_rejects": "mean_recovery_tube_interaction_root_unrecoverable_rejects_on_attempts",
        "joint_incompatibility_rejects": "mean_recovery_tube_interaction_joint_incompatibility_rejects_on_attempts",
        "environment_compatibility_checks": "mean_recovery_tube_interaction_environment_compatibility_checks_on_attempts",
        "environment_compatibility_rejects": "mean_recovery_tube_interaction_environment_compatibility_rejects_on_attempts",
        "joint_compatibility_checks": "mean_recovery_tube_interaction_joint_compatibility_checks_on_attempts",
        "joint_compatibility_rejects": "mean_recovery_tube_interaction_joint_compatibility_rejects_on_attempts",
        "joint_assignment_backtracks": "mean_recovery_tube_interaction_joint_assignment_backtracks_on_attempts",
    }
    attempt_means: dict[str, float | None] = {}
    attempt_totals: dict[str, float] = {}
    for label, key in attempt_keys.items():
        total, den = _weighted_total(rows, ids, key, attempts)
        attempt_totals[label] = float(total)
        attempt_means[label] = float(total / den) if den else None

    selected_den: dict[str, int] = {}
    for sid in ids:
        selected_den[sid] = _count_from_rate(
            rows, [sid], "recovery_tube_interaction_selection_rate_on_certified_steps", certificates
        )
    selected_keys = {
        "blockers": "mean_recovery_tube_interaction_selected_blockers_on_interaction_steps",
        "roots": "mean_recovery_tube_interaction_selected_roots_on_interaction_steps",
        "minimum_root_mass": "mean_recovery_tube_interaction_selected_minimum_root_mass_on_interaction_steps",
        "maximum_response_burden": "mean_recovery_tube_interaction_selected_maximum_response_burden_on_interaction_steps",
        "profile_evaluations": "mean_recovery_tube_interaction_selected_profile_evaluations_on_interaction_steps",
        "environment_agents": "mean_recovery_tube_interaction_selected_environment_agents_on_interaction_steps",
        "environment_checks": "mean_recovery_tube_interaction_selected_environment_checks_on_interaction_steps",
    }
    selected_means: dict[str, float | None] = {}
    for label, key in selected_keys.items():
        total, den = _weighted_total(rows, ids, key, selected_den)
        selected_means[label] = float(total / den) if den else None

    fallback_delta_sum, fallback_delta_den = _weighted_total(
        rows, ids, "mean_recovery_tube_selected_fallback_score_delta_on_action_changes", changes
    )
    ready = float(attempt_totals.get("support_agents_ready", 0.0))
    support = float(attempt_totals.get("support_agents_total", 0.0))
    retained_roots = float(attempt_totals.get("retained_roots", 0.0))
    eligible_profiles = float(attempt_totals.get("eligible_profiles", 0.0))

    return {
        "aggregation": "step/attempt/certificate weighted pooled reconstruction from scenario diagnostics",
        "total_policy_steps": total_steps,
        "total_probe_steps": total_probes,
        "total_certificate_steps": total_certificates,
        "total_action_change_steps": total_changes,
        "total_interaction_attempt_steps": total_attempts,
        "probe_step_rate": float(total_probes / total_steps) if total_steps else None,
        "certificate_step_rate": float(total_certificates / total_steps) if total_steps else None,
        "certificate_rate_on_probes": float(total_certificates / total_probes) if total_probes else None,
        "action_change_step_rate": float(total_changes / total_steps) if total_steps else None,
        "nested_v39_selected_certificate_steps": nested_selected,
        "nested_v39_selection_rate_on_certified_steps": (
            float(nested_selected / total_certificates) if total_certificates else None
        ),
        "interaction_selected_certificate_steps": interaction_selected,
        "interaction_selection_rate_on_certified_steps": (
            float(interaction_selected / total_certificates) if total_certificates else None
        ),
        "interaction_completion_rate_on_attempts": (
            float(interaction_selected / total_attempts) if total_attempts else None
        ),
        "interaction_attempt_step_rate": (
            float(total_attempts / total_steps) if total_steps else None
        ),
        "lifted_selected_certificate_steps": lifted_selected,
        "event_release_selected_certificate_steps": event_selected,
        "attempt_weighted_means": attempt_means,
        "attempt_weighted_totals": attempt_totals,
        "support_agent_readiness_rate": float(ready / support) if support > 0.0 else None,
        "eligible_profiles_per_retained_root": (
            float(eligible_profiles / retained_roots) if retained_roots > 0.0 else None
        ),
        "interaction_selected_weighted_means": selected_means,
        "mean_fallback_score_delta_on_action_changes": (
            float(fallback_delta_sum / fallback_delta_den) if fallback_delta_den else None
        ),
        "mechanism_nonzero": bool(interaction_selected > 0),
    }



def _pooled_blocker_query_mechanism(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    steps: dict[str, int] = {}
    certificates: dict[str, int] = {}
    attempts: dict[str, int] = {}
    for sid in ids:
        diag = rows[sid].get("diagnostics", {}) or {}
        n = int(diag.get("steps", 0) or 0)
        steps[sid] = n
        certificates[sid] = int(round(float(diag.get("recovery_tube_certificate_step_rate", 0.0) or 0.0) * n))
        attempts[sid] = int(round(float(diag.get("recovery_tube_blocker_query_attempt_step_rate", 0.0) or 0.0) * n))
    total_steps = int(sum(steps.values()))
    total_certificates = int(sum(certificates.values()))
    total_attempts = int(sum(attempts.values()))
    query_selected = _count_from_rate(
        rows, ids, "recovery_tube_blocker_query_selection_rate_on_certified_steps", certificates
    )
    nested_v42_selected = _count_from_rate(
        rows, ids, "recovery_tube_nested_v42_selection_rate_on_certified_steps", certificates
    )
    mean_keys = {
        "query_agents": "mean_recovery_tube_blocker_query_agents_on_attempts",
        "query_ready_agents": "mean_recovery_tube_blocker_query_ready_agents_on_attempts",
        "hypotheses_evaluated": "mean_recovery_tube_blocker_query_hypotheses_on_attempts",
        "unsupported_blocker_rejects": "mean_recovery_tube_blocker_query_unsupported_rejects_on_attempts",
        "root_unrecoverable_rejects": "mean_recovery_tube_blocker_query_root_unrecoverable_rejects_on_attempts",
        "environment_cache_hits": "mean_recovery_tube_blocker_query_environment_cache_hits_on_attempts",
        "joint_cache_hits": "mean_recovery_tube_blocker_query_joint_cache_hits_on_attempts",
        "successor_context_cache_hits": "mean_recovery_tube_blocker_query_successor_context_cache_hits_on_attempts",
    }
    totals: dict[str, float] = {}
    means: dict[str, float | None] = {}
    for label, key in mean_keys.items():
        total, den = _weighted_total(rows, ids, key, attempts)
        totals[label] = float(total)
        means[label] = float(total / den) if den else None
    return {
        "aggregation": "step/attempt/certificate weighted pooled reconstruction from scenario diagnostics",
        "total_policy_steps": total_steps,
        "total_certificate_steps": total_certificates,
        "total_blocker_query_attempt_steps": total_attempts,
        "total_blocker_query_selected_certificate_steps": query_selected,
        "total_nested_v42_selected_certificate_steps": nested_v42_selected,
        "blocker_query_attempt_step_rate": float(total_attempts / total_steps) if total_steps else None,
        "blocker_query_selection_rate_on_certified_steps": float(query_selected / total_certificates) if total_certificates else None,
        "blocker_query_completion_rate_on_attempts": float(query_selected / total_attempts) if total_attempts else None,
        "nested_v42_selection_rate_on_certified_steps": float(nested_v42_selected / total_certificates) if total_certificates else None,
        "attempt_weighted_means": means,
        "attempt_weighted_totals": totals,
        "late_bound_query_mechanism_nonzero": bool(query_selected > 0),
        "performance_cache_hits_are_diagnostic_only": True,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cowp", required=True)
    parser.add_argument("--rvr")
    parser.add_argument("--v33-rosh")
    parser.add_argument("--v35-control-projected-spectrum")
    parser.add_argument("--v36-recovery-frontier")
    parser.add_argument("--v37-recourse-returnability-bridge")
    parser.add_argument("--v38-shift-closed-tube")
    parser.add_argument("--v39-conflict-window-tube")
    parser.add_argument("--root-conditioned-control-reachable-responder-support", required=True)
    parser.add_argument("--v42-interaction-aware-reachable-response-envelope")
    parser.add_argument("--development-selected", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["counterfactual48", "fresh37", "exact200"],
        default="counterfactual48",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = {
        "cowp": _load(args.cowp),
        "rvr": _load(args.rvr),
        "v33_rosh": _load(args.v33_rosh),
        "v35_control_projected_spectrum": _load(args.v35_control_projected_spectrum),
        "v36_recovery_frontier": _load(args.v36_recovery_frontier),
        "v37_recourse_returnability_bridge": _load(args.v37_recourse_returnability_bridge),
        "v38_shift_closed_tube": _load(args.v38_shift_closed_tube),
        "v39_conflict_window_tube": _load(args.v39_conflict_window_tube),
        "v42_interaction_aware_reachable_response_envelope": _load(args.v42_interaction_aware_reachable_response_envelope),
        METHOD_KEY: _load(args.root_conditioned_control_reachable_responder_support),
    }
    data = {name: value for name, value in data.items() if value is not None}
    rowmap = {name: _rows(value) for name, value in data.items()}
    ids = list(rowmap["cowp"].keys())
    idset = set(ids)
    mismatch = {
        name: sorted(idset.symmetric_difference(set(rows)))
        for name, rows in rowmap.items()
        if set(rows) != idset
    }
    if mismatch:
        raise SystemExit(f"scenario set mismatch: {mismatch}")

    out: dict[str, Any] = {
        "schema_version": "cowp_root_conditioned_control_reachable_responder_support_analysis_v1",
        "canonical_method": CANONICAL_METHOD,
        "development_selected": bool(args.development_selected),
        "paper_evidence": False,
        "scenario_count": len(ids),
        "stage": str(args.stage),
        "methods": {},
        "paired_vs_cowp": {},
    }
    for name, rows in rowmap.items():
        out["methods"][name] = _aggregate(rows, ids)
        if name != "cowp":
            out["paired_vs_cowp"][name] = _paired(rowmap["cowp"], rows, ids)

    if "rvr" in rowmap:
        old_rescued = set(out["paired_vs_cowp"]["rvr"]["Collision"]["rescued_ids"])
        old_induced = set(out["paired_vs_cowp"]["rvr"]["Collision"]["induced_ids"])
        out["rvr_counterexample_retention"] = {
            METHOD_KEY: {
                "old_rvr_rescues_total": len(old_rescued),
                "old_rvr_rescues_retained": int(
                    sum(not _event(rowmap[METHOD_KEY][sid], "CollisionRate") for sid in old_rescued)
                ),
                "old_rvr_induced_total": len(old_induced),
                "old_rvr_induced_avoided": int(
                    sum(not _event(rowmap[METHOD_KEY][sid], "CollisionRate") for sid in old_induced)
                ),
            }
        }

    for previous in (
        "v33_rosh",
        "v35_control_projected_spectrum",
        "v36_recovery_frontier",
        "v37_recourse_returnability_bridge",
        "v38_shift_closed_tube",
        "v39_conflict_window_tube",
        "v42_interaction_aware_reachable_response_envelope",
    ):
        if previous in rowmap:
            out[f"paired_vs_{previous}"] = _paired(rowmap[previous], rowmap[METHOD_KEY], ids)

    paired = out["paired_vs_cowp"][METHOD_KEY]
    ep_delta = paired["EP"]["delta_mean"]
    pooled = _pooled_interaction_mechanism(rowmap[METHOD_KEY], ids)
    blocker_query_pooled = _pooled_blocker_query_mechanism(rowmap[METHOD_KEY], ids)
    switch_rate = pooled.get("action_change_step_rate")

    if args.stage == "counterfactual48":
        retention = out["rvr_counterexample_retention"][METHOD_KEY]
        checks = {
            "retain_at_least_5_of_10_old_rvr_rescues": retention["old_rvr_rescues_retained"] >= 5,
            "avoid_at_least_7_of_9_old_rvr_induced": retention["old_rvr_induced_avoided"] >= 7,
            "net_remove_at_least_3_cowp_collisions": paired["Collision"]["net_failures_removed"] >= 3,
            "kinematics_net_regression_at_most_1_scene": paired["Kinematics"]["net_failures_removed"] >= -1,
            "mean_ep_delta_not_below_minus_0_05": ep_delta is not None and ep_delta >= -0.05,
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    elif args.stage == "fresh37":
        checks = {
            "no_net_collision_harm": paired["Collision"]["net_failures_removed"] >= 0,
            "no_net_cr_harm": paired["CR"]["net_failures_removed"] >= 0,
            "offroad_net_regression_at_most_1_scene": paired["Offroad"]["net_failures_removed"] >= -1,
            "kinematics_net_regression_at_most_1_scene": paired["Kinematics"]["net_failures_removed"] >= -1,
            "mean_ep_delta_not_below_minus_0_03": ep_delta is not None and ep_delta >= -0.03,
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    else:
        checks = {"development_confirmation_only": True}

    out["preregistered_gate"] = {
        METHOD_KEY: {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "paper_evidence": False,
        }
    }
    out["root_conditioned_control_reachable_responder_support_mechanism_diagnostic"] = {
        **pooled,
        "blocker_conditioned_query": blocker_query_pooled,
        "support_diagnostics_are_attribution_not_gate": True,
        "interaction_only_selection_required_for_specific_mechanism_attribution": True,
        "interpretation": (
            "The inherited six-item outcome conjunction gate is unchanged. V16.8.43 exact-nests V16.8.42 first; only a V42-empty recovery step may add late-bound natural-root support for agents in the frozen collision context. The V39 tube remains nested inside V42. High-mass natural-root "
            "coverage, same-root low-burden current/shift responses, residual hard "
            "physical checks, and cross-agent joint compatibility are mechanism "
            "diagnostics. They do not replace or weaken the frozen promotion gate."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
