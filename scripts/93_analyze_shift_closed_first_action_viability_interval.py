from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: str | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text()) if path else None


def _rows(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["scenario_id"]): r for r in d.get("scenario_results", [])}


def _metric(row: dict[str, Any], key: str) -> float:
    value = (row.get("standard_metrics", {}) or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("nan")


def _event(row: dict[str, Any], key: str) -> bool:
    value = _metric(row, key)
    return bool(math.isfinite(value) and value > 0.0)


def _mcnemar_exact(rescued: int, induced: int) -> float:
    n = int(rescued + induced)
    if n <= 0:
        return 1.0
    k = min(int(rescued), int(induced))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_mean_ci(
    x: np.ndarray,
    *,
    seed: int = 16840,
    draws: int = 10_000,
) -> list[float] | None:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        values[i] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _diag_value(row: dict[str, Any], key: str) -> float | None:
    value = (row.get("diagnostics", {}) or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _aggregate(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    event_keys = {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }
    out: dict[str, Any] = {"n": len(ids)}
    for label, key in event_keys.items():
        out[label] = float(np.mean([_event(rows[s], key) for s in ids])) if ids else 0.0
    ep = np.asarray([_metric(rows[s], "EP") for s in ids], dtype=np.float64)
    ep = ep[np.isfinite(ep)]
    out["EP"] = float(ep.mean()) if ep.size else None

    # These are explicitly scenario-macro averages. Mechanism attribution below
    # separately reconstructs probe/certificate-weighted pooled quantities, avoiding
    # the V38 reporting ambiguity in which a scenario with one probe and a scenario
    # with eighty probes received the same weight.
    diag_keys = [
        "fallback_step_rate",
        "zero_conventional_candidate_step_rate",
        "mean_conventional_candidates",
        "mean_valid_candidates",
        "mean_max_collision_safe_prefix_steps",
        "mean_selected_collision_safe_prefix_steps",
        "recovery_switch_step_rate",
        "selected_waymax_kinematic_feasible_step_rate",
        "selected_waymax_kinematic_feasible_rate_on_recovery_switch_steps",
        "shift_closed_first_action_viability_interval_step_rate",
        "recovery_tube_probe_step_rate",
        "recovery_tube_certificate_step_rate",
        "recovery_tube_action_change_step_rate",
        "recovery_tube_lifted_selection_rate_on_certified_steps",
        "recovery_tube_event_release_selection_rate_on_certified_steps",
        "recovery_tube_nested_v39_selection_rate_on_certified_steps",
        "recovery_tube_first_action_interval_attempt_step_rate",
        "recovery_tube_first_action_interval_selection_rate_on_certified_steps",
        "recovery_tube_new_first_action_selection_rate_on_certified_steps",
        "mean_recovery_tube_first_action_interval_basis_count_on_attempts",
        "mean_recovery_tube_first_action_interval_seed_evaluations_on_attempts",
        "mean_recovery_tube_first_action_interval_boundary_proposals_on_attempts",
        "mean_recovery_tube_first_action_interval_hypotheses_on_attempts",
        "mean_recovery_tube_first_action_interval_unique_actions_on_attempts",
        "mean_recovery_tube_first_action_interval_new_actions_on_attempts",
        "mean_recovery_tube_first_action_interval_full_safe_on_attempts",
        "mean_recovery_tube_first_action_interval_shift_closed_on_attempts",
        "mean_recovery_tube_first_action_interval_only_parent_count_on_attempts",
        "mean_recovery_tube_selected_first_accel_fraction_on_interval_steps",
        "mean_recovery_tube_parent_pool_on_probes",
        "mean_recovery_tube_parent_action_classes_on_probes",
        "mean_recovery_tube_parents_with_nominal_conflict_on_probes",
        "mean_recovery_tube_parent_first_conflict_step_on_probes",
        "mean_recovery_tube_parent_last_conflict_step_on_probes",
        "mean_recovery_tube_hypotheses_generated_on_probes",
        "mean_recovery_tube_unique_action_hypotheses_on_probes",
        "mean_recovery_tube_full_physically_safe_on_probes",
        "mean_recovery_tube_shift_closed_on_probes",
        "mean_recovery_tube_nominal_shift_closed_on_probes",
        "mean_recovery_tube_lower_envelope_shift_closed_on_probes",
        "mean_recovery_tube_upper_envelope_shift_closed_on_probes",
        "mean_recovery_tube_event_release_shift_closed_on_probes",
        "mean_recovery_tube_lower_event_release_shift_closed_on_probes",
        "mean_recovery_tube_upper_event_release_shift_closed_on_probes",
        "mean_recovery_tube_lifted_only_parent_count_on_probes",
        "mean_recovery_tube_event_release_only_parent_count_on_probes",
        "mean_recovery_tube_abs_first_accel_delta_on_certified_steps",
        "mean_recovery_tube_selected_release_edge_on_certified_steps",
        "mean_recovery_tube_selected_nonnominal_edges_on_certified_steps",
        "mean_recovery_tube_selected_collision_margin_on_certified_steps",
        "mean_recovery_tube_selected_shift_collision_margin_on_certified_steps",
        "mean_recovery_tube_selected_fallback_score_delta_on_action_changes",
    ]
    macro: dict[str, float | None] = {}
    for key in diag_keys:
        values = [_diag_value(rows[s], key) for s in ids]
        finite = [v for v in values if v is not None]
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
        rescued = [s for s in ids if _event(base[s], key) and not _event(alt[s], key)]
        induced = [s for s in ids if not _event(base[s], key) and _event(alt[s], key)]
        shared = [s for s in ids if _event(base[s], key) and _event(alt[s], key)]
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
    for s in ids:
        a, b = _metric(base[s], "EP"), _metric(alt[s], "EP")
        if math.isfinite(a) and math.isfinite(b):
            delta = float(b - a)
            deltas.append(delta)
            per_scene[s] = delta
    arr = np.asarray(deltas, dtype=np.float64)
    out["EP"] = {
        "paired_n": int(arr.size),
        "delta_mean": float(arr.mean()) if arr.size else None,
        "delta_median": float(np.median(arr)) if arr.size else None,
        "bootstrap95": _bootstrap_mean_ci(arr),
        "per_scene_delta": per_scene,
    }
    return out


def _weighted_total(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
    mean_key: str,
    denominator_counts: dict[str, int],
) -> tuple[float, int]:
    total = 0.0
    denominator = 0
    for s in ids:
        n = int(denominator_counts.get(s, 0))
        if n <= 0:
            continue
        value = _diag_value(rows[s], mean_key)
        if value is None:
            continue
        total += float(value) * n
        denominator += n
    return total, denominator


def _pooled_tube_mechanism(
    rows: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    steps: dict[str, int] = {}
    probes: dict[str, int] = {}
    certificates: dict[str, int] = {}
    changes: dict[str, int] = {}
    interval_attempts: dict[str, int] = {}
    for s in ids:
        diag = rows[s].get("diagnostics", {}) or {}
        n = int(diag.get("steps", 0) or 0)
        steps[s] = n
        probes[s] = int(round(float(diag.get("recovery_tube_probe_step_rate", 0.0) or 0.0) * n))
        certificates[s] = int(round(float(diag.get("recovery_tube_certificate_step_rate", 0.0) or 0.0) * n))
        changes[s] = int(round(float(diag.get("recovery_tube_action_change_step_rate", 0.0) or 0.0) * n))
        interval_attempts[s] = int(round(float(
            diag.get("recovery_tube_first_action_interval_attempt_step_rate", 0.0) or 0.0
        ) * n))

    total_steps = int(sum(steps.values()))
    total_probes = int(sum(probes.values()))
    total_certificates = int(sum(certificates.values()))
    total_changes = int(sum(changes.values()))
    total_interval_attempts = int(sum(interval_attempts.values()))

    def conditional_count(rate_key: str, den: dict[str, int]) -> int:
        total = 0
        for s in ids:
            n = int(den.get(s, 0))
            if n <= 0:
                continue
            value = _diag_value(rows[s], rate_key)
            if value is not None:
                total += int(round(value * n))
        return int(total)

    lifted_selected = conditional_count(
        "recovery_tube_lifted_selection_rate_on_certified_steps", certificates
    )
    event_selected = conditional_count(
        "recovery_tube_event_release_selection_rate_on_certified_steps", certificates
    )
    nested_v39_selected = conditional_count(
        "recovery_tube_nested_v39_selection_rate_on_certified_steps", certificates
    )
    interval_selected = conditional_count(
        "recovery_tube_first_action_interval_selection_rate_on_certified_steps",
        certificates,
    )
    new_first_action_selected = conditional_count(
        "recovery_tube_new_first_action_selection_rate_on_certified_steps",
        certificates,
    )

    probe_mean_keys = {
        "parent_pool": "mean_recovery_tube_parent_pool_on_probes",
        "parent_action_classes": "mean_recovery_tube_parent_action_classes_on_probes",
        "parents_with_nominal_conflict": "mean_recovery_tube_parents_with_nominal_conflict_on_probes",
        "parent_first_conflict_step": "mean_recovery_tube_parent_first_conflict_step_on_probes",
        "parent_last_conflict_step": "mean_recovery_tube_parent_last_conflict_step_on_probes",
        "hypotheses_generated": "mean_recovery_tube_hypotheses_generated_on_probes",
        "unique_first_actions": "mean_recovery_tube_unique_action_hypotheses_on_probes",
        "full_physically_safe_witnesses": "mean_recovery_tube_full_physically_safe_on_probes",
        "shift_closed_witnesses": "mean_recovery_tube_shift_closed_on_probes",
        "nominal_shift_closed_witnesses": "mean_recovery_tube_nominal_shift_closed_on_probes",
        "lower_all_shift_closed_witnesses": "mean_recovery_tube_lower_envelope_shift_closed_on_probes",
        "upper_all_shift_closed_witnesses": "mean_recovery_tube_upper_envelope_shift_closed_on_probes",
        "event_release_shift_closed_witnesses": "mean_recovery_tube_event_release_shift_closed_on_probes",
        "lower_event_release_shift_closed_witnesses": "mean_recovery_tube_lower_event_release_shift_closed_on_probes",
        "upper_event_release_shift_closed_witnesses": "mean_recovery_tube_upper_event_release_shift_closed_on_probes",
        "lifted_only_parent_support": "mean_recovery_tube_lifted_only_parent_count_on_probes",
        "event_release_only_parent_support": "mean_recovery_tube_event_release_only_parent_count_on_probes",
    }
    pooled_probe: dict[str, Any] = {}
    totals: dict[str, float] = {}
    for label, key in probe_mean_keys.items():
        total, den = _weighted_total(rows, ids, key, probes)
        totals[label] = float(total)
        pooled_probe[label] = float(total / den) if den else None

    interval_mean_keys = {
        "basis_count": "mean_recovery_tube_first_action_interval_basis_count_on_attempts",
        "seed_evaluations": "mean_recovery_tube_first_action_interval_seed_evaluations_on_attempts",
        "boundary_proposals": "mean_recovery_tube_first_action_interval_boundary_proposals_on_attempts",
        "hypotheses_evaluated": "mean_recovery_tube_first_action_interval_hypotheses_on_attempts",
        "unique_certified_actions": "mean_recovery_tube_first_action_interval_unique_actions_on_attempts",
        "new_certified_first_actions": "mean_recovery_tube_first_action_interval_new_actions_on_attempts",
        "full_physically_safe": "mean_recovery_tube_first_action_interval_full_safe_on_attempts",
        "shift_closed": "mean_recovery_tube_first_action_interval_shift_closed_on_attempts",
        "interval_only_parent_support": "mean_recovery_tube_first_action_interval_only_parent_count_on_attempts",
    }
    pooled_interval: dict[str, Any] = {}
    interval_totals: dict[str, float] = {}
    for label, key in interval_mean_keys.items():
        total, den = _weighted_total(rows, ids, key, interval_attempts)
        interval_totals[label] = float(total)
        pooled_interval[label] = float(total / den) if den else None

    cert_mean_keys = {
        "abs_first_accel_delta": "mean_recovery_tube_abs_first_accel_delta_on_certified_steps",
        "selected_release_edge": "mean_recovery_tube_selected_release_edge_on_certified_steps",
        "selected_nonnominal_edges": "mean_recovery_tube_selected_nonnominal_edges_on_certified_steps",
        "selected_collision_margin": "mean_recovery_tube_selected_collision_margin_on_certified_steps",
        "selected_shift_collision_margin": "mean_recovery_tube_selected_shift_collision_margin_on_certified_steps",
    }
    pooled_cert: dict[str, Any] = {}
    for label, key in cert_mean_keys.items():
        total, den = _weighted_total(rows, ids, key, certificates)
        pooled_cert[label] = float(total / den) if den else None

    fallback_delta_sum, fallback_delta_den = _weighted_total(
        rows,
        ids,
        "mean_recovery_tube_selected_fallback_score_delta_on_action_changes",
        changes,
    )
    full_witnesses = float(totals.get("full_physically_safe_witnesses", 0.0))
    shifted_witnesses = float(totals.get("shift_closed_witnesses", 0.0))
    interval_selected_den = {
        s: int(round(float(
            (rows[s].get("diagnostics", {}) or {}).get(
                "recovery_tube_first_action_interval_selection_rate_on_certified_steps",
                0.0,
            ) or 0.0
        ) * certificates[s]))
        for s in ids
    }
    fraction_sum, fraction_den = _weighted_total(
        rows,
        ids,
        "mean_recovery_tube_selected_first_accel_fraction_on_interval_steps",
        interval_selected_den,
    )

    return {
        "aggregation": "probe/certificate/action-change weighted pooled reconstruction from scenario diagnostics",
        "total_policy_steps": total_steps,
        "total_probe_steps": total_probes,
        "total_certificate_steps": total_certificates,
        "total_action_change_steps": total_changes,
        "total_interval_attempt_steps": total_interval_attempts,
        "probe_step_rate": float(total_probes / total_steps) if total_steps else None,
        "certificate_step_rate": float(total_certificates / total_steps) if total_steps else None,
        "certificate_rate_on_probes": float(total_certificates / total_probes) if total_probes else None,
        "action_change_step_rate": float(total_changes / total_steps) if total_steps else None,
        "lifted_selected_certificate_steps": lifted_selected,
        "lifted_selection_rate_on_certified_steps": (
            float(lifted_selected / total_certificates) if total_certificates else None
        ),
        "event_release_selected_certificate_steps": event_selected,
        "event_release_selection_rate_on_certified_steps": (
            float(event_selected / total_certificates) if total_certificates else None
        ),
        "nested_v39_selected_certificate_steps": nested_v39_selected,
        "nested_v39_selection_rate_on_certified_steps": (
            float(nested_v39_selected / total_certificates) if total_certificates else None
        ),
        "interval_selected_certificate_steps": interval_selected,
        "interval_selection_rate_on_certified_steps": (
            float(interval_selected / total_certificates) if total_certificates else None
        ),
        "new_first_action_selected_certificate_steps": new_first_action_selected,
        "new_first_action_selection_rate_on_certified_steps": (
            float(new_first_action_selected / total_certificates) if total_certificates else None
        ),
        "interval_attempt_step_rate": (
            float(total_interval_attempts / total_steps) if total_steps else None
        ),
        "interval_completion_rate_on_attempts": (
            float(interval_selected / total_interval_attempts)
            if total_interval_attempts else None
        ),
        "probe_weighted_means": pooled_probe,
        "interval_attempt_weighted_means": pooled_interval,
        "interval_attempt_weighted_totals": interval_totals,
        "mean_selected_first_accel_fraction_on_interval_steps": (
            float(fraction_sum / fraction_den) if fraction_den else None
        ),
        "probe_weighted_totals": totals,
        "certificate_weighted_means": pooled_cert,
        "mean_fallback_score_delta_on_action_changes": (
            float(fallback_delta_sum / fallback_delta_den) if fallback_delta_den else None
        ),
        "shift_closure_retention_of_full_safe_witnesses": (
            float(shifted_witnesses / full_witnesses) if full_witnesses > 0 else None
        ),
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
    parser.add_argument("--shift-closed-first-action-viability-interval", required=True)
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
        "shift_closed_first_action_viability_interval": _load(
            args.shift_closed_first_action_viability_interval
        ),
    }
    data = {k: v for k, v in data.items() if v is not None}
    rowmap = {k: _rows(v) for k, v in data.items()}
    ids = list(rowmap["cowp"].keys())
    idset = set(ids)
    mismatch = {
        k: sorted(idset.symmetric_difference(set(rows)))
        for k, rows in rowmap.items()
        if set(rows) != idset
    }
    if mismatch:
        raise SystemExit(f"scenario set mismatch: {mismatch}")

    out: dict[str, Any] = {
        "schema_version": "cowp_shift_closed_first_action_viability_interval_analysis_v1",
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

    name = "shift_closed_first_action_viability_interval"
    if "rvr" in rowmap:
        old_rescued = set(out["paired_vs_cowp"]["rvr"]["Collision"]["rescued_ids"])
        old_induced = set(out["paired_vs_cowp"]["rvr"]["Collision"]["induced_ids"])
        out["rvr_counterexample_retention"] = {
            name: {
                "old_rvr_rescues_total": len(old_rescued),
                "old_rvr_rescues_retained": int(
                    sum(not _event(rowmap[name][s], "CollisionRate") for s in old_rescued)
                ),
                "old_rvr_induced_total": len(old_induced),
                "old_rvr_induced_avoided": int(
                    sum(not _event(rowmap[name][s], "CollisionRate") for s in old_induced)
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
    ):
        if previous in rowmap:
            out[f"paired_vs_{previous}"] = _paired(rowmap[previous], rowmap[name], ids)

    paired = out["paired_vs_cowp"][name]
    ep_delta = paired["EP"]["delta_mean"]
    pooled = _pooled_tube_mechanism(rowmap[name], ids)
    switch_rate = pooled.get("action_change_step_rate")

    if args.stage == "counterfactual48":
        retention = out["rvr_counterexample_retention"][name]
        checks = {
            "retain_at_least_5_of_10_old_rvr_rescues": (
                retention["old_rvr_rescues_retained"] >= 5
            ),
            "avoid_at_least_7_of_9_old_rvr_induced": (
                retention["old_rvr_induced_avoided"] >= 7
            ),
            "net_remove_at_least_3_cowp_collisions": (
                paired["Collision"]["net_failures_removed"] >= 3
            ),
            "kinematics_net_regression_at_most_1_scene": (
                paired["Kinematics"]["net_failures_removed"] >= -1
            ),
            "mean_ep_delta_not_below_minus_0_05": (
                ep_delta is not None and ep_delta >= -0.05
            ),
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    elif args.stage == "fresh37":
        checks = {
            "no_net_collision_harm": paired["Collision"]["net_failures_removed"] >= 0,
            "no_net_cr_harm": paired["CR"]["net_failures_removed"] >= 0,
            "offroad_net_regression_at_most_1_scene": (
                paired["Offroad"]["net_failures_removed"] >= -1
            ),
            "kinematics_net_regression_at_most_1_scene": (
                paired["Kinematics"]["net_failures_removed"] >= -1
            ),
            "mean_ep_delta_not_below_minus_0_03": (
                ep_delta is not None and ep_delta >= -0.03
            ),
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    else:
        checks = {"development_confirmation_only": True}

    out["preregistered_gate"] = {
        name: {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "paper_evidence": False,
        }
    }
    out["first_action_viability_interval_mechanism_diagnostic"] = {
        **pooled,
        "new_first_action_support_is_attribution_not_gate": True,
        "interpretation": (
            "The inherited six-item outcome gate is unchanged. Interval-only/new-first-"
            "action support and actual new-first-action selection are required to "
            "attribute any gain specifically to V40 first-action interval completion, "
            "but are not post-hoc additional promotion thresholds."
        ),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
