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
    v = (row.get("standard_metrics", {}) or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else float("nan")


def _event(row: dict[str, Any], key: str) -> bool:
    v = _metric(row, key)
    return bool(math.isfinite(v) and v > 0.0)


def _mcnemar_exact(rescued: int, induced: int) -> float:
    n = int(rescued + induced)
    if n <= 0:
        return 1.0
    k = min(int(rescued), int(induced))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_mean_ci(x: np.ndarray, seed: int = 16838, draws: int = 10000) -> list[float] | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        vals[i] = float(rng.choice(x, size=x.size, replace=True).mean())
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _aggregate(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    event_keys = {"CR": "CR", "Collision": "CollisionRate", "Offroad": "OffroadRate", "Kinematics": "KinematicsInfeasibilityRate"}
    out: dict[str, Any] = {"n": len(ids)}
    for label, key in event_keys.items():
        out[label] = float(np.mean([_event(rows[s], key) for s in ids])) if ids else 0.0
    ep = np.asarray([_metric(rows[s], "EP") for s in ids], dtype=np.float64)
    ep = ep[np.isfinite(ep)]
    out["EP"] = float(ep.mean()) if ep.size else None
    diag_keys = [
        "fallback_step_rate", "zero_conventional_candidate_step_rate", "mean_conventional_candidates", "mean_valid_candidates",
        "mean_max_collision_safe_prefix_steps", "mean_selected_collision_safe_prefix_steps", "recovery_switch_step_rate",
        "recovery_hysteresis_active_step_rate", "recovery_hysteresis_entry_step_rate", "recovery_hysteresis_continue_step_rate",
        "recovery_hysteresis_exit_step_rate", "recovery_hysteresis_clear_step_rate",
        "recourse_returnability_bridge_step_rate", "recourse_returnability_probe_step_rate",
        "recourse_returnability_strict_dominance_rate_on_probes",
        "recourse_base_direct_restore_rate_on_probes", "recourse_rvr_direct_restore_rate_on_probes",
        "mean_recourse_base_macro_count_on_probes", "mean_recourse_rvr_macro_count_on_probes",
        "recourse_current_action_survival_step_rate",
        "mean_recourse_base_action_classes_available_on_probes", "mean_recourse_rvr_action_classes_available_on_probes",
        "mean_recourse_base_action_classes_evaluated_on_probes", "mean_recourse_rvr_action_classes_evaluated_on_probes",
        "recovery_bridge_pending_step_rate", "recovery_bridge_entry_step_rate",
        "recovery_bridge_direct_entry_step_rate", "recovery_bridge_recourse_execution_step_rate",
        "recovery_bridge_abort_step_rate", "mean_direct_restoring_candidate_count_on_bridge_steps",
        "mean_recovery_bridge_allowed_macro_count_on_bridge_steps",
        "mean_recourse_bridge_candidate_pool_on_bridge_steps",
        "mean_recourse_bridge_action_classes_available_on_bridge_steps",
        "mean_recourse_bridge_action_classes_evaluated_on_bridge_steps",
        "mean_recourse_bridge_minimum_prefix_steps_on_bridge_steps",
        "recovery_bridge_execution_rate_on_bridge_steps", "recovery_bridge_abort_rate_on_bridge_steps",
        "selected_waymax_kinematic_feasible_step_rate", "selected_waymax_kinematic_feasible_rate_on_recovery_switch_steps",
        "shift_closed_control_reachable_tube_step_rate", "recovery_tube_probe_step_rate",
        "recovery_tube_certificate_step_rate", "recovery_tube_action_change_step_rate",
        "recovery_tube_lifted_selection_rate_on_certified_steps",
        "mean_recovery_tube_parent_pool_on_probes", "mean_recovery_tube_parent_action_classes_on_probes",
        "mean_recovery_tube_hypotheses_generated_on_probes", "mean_recovery_tube_unique_action_hypotheses_on_probes",
        "mean_recovery_tube_full_physically_safe_on_probes", "mean_recovery_tube_shift_closed_on_probes",
        "mean_recovery_tube_nominal_shift_closed_on_probes",
        "mean_recovery_tube_lower_envelope_shift_closed_on_probes",
        "mean_recovery_tube_upper_envelope_shift_closed_on_probes",
        "mean_recovery_tube_lifted_only_parent_count_on_probes",
        "mean_recovery_tube_abs_first_accel_delta_on_certified_steps",
        "mean_recovery_tube_selected_collision_margin_on_certified_steps",
        "mean_recovery_tube_selected_shift_collision_margin_on_certified_steps",
        "mean_recovery_tube_selected_fallback_score_delta_on_action_changes",
    ]
    for key in diag_keys:
        vals = []
        for s in ids:
            v = (rows[s].get("diagnostics", {}) or {}).get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vals.append(float(v))
        out[key] = float(np.mean(vals)) if vals else None
    return out


def _paired(base: dict[str, dict[str, Any]], alt: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, key in {"CR": "CR", "Collision": "CollisionRate", "Offroad": "OffroadRate", "Kinematics": "KinematicsInfeasibilityRate"}.items():
        rescued = [s for s in ids if _event(base[s], key) and not _event(alt[s], key)]
        induced = [s for s in ids if not _event(base[s], key) and _event(alt[s], key)]
        shared = [s for s in ids if _event(base[s], key) and _event(alt[s], key)]
        out[label] = {
            "rescued": len(rescued), "induced": len(induced), "shared_failure": len(shared),
            "net_failures_removed": len(rescued) - len(induced), "mcnemar_exact_p": _mcnemar_exact(len(rescued), len(induced)),
            "rescued_ids": rescued, "induced_ids": induced,
        }
    dep = []
    for s in ids:
        a, b = _metric(base[s], "EP"), _metric(alt[s], "EP")
        if math.isfinite(a) and math.isfinite(b): dep.append(b - a)
    arr = np.asarray(dep, dtype=np.float64)
    out["EP"] = {"paired_n": int(arr.size), "delta_mean": float(arr.mean()) if arr.size else None,
                 "delta_median": float(np.median(arr)) if arr.size else None, "bootstrap95": _bootstrap_mean_ci(arr)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--rvr")
    ap.add_argument("--v33-rosh")
    ap.add_argument("--v35-control-projected-spectrum")
    ap.add_argument("--v36-recovery-frontier")
    ap.add_argument("--v37-recourse-returnability-bridge")
    ap.add_argument("--shift-closed-control-reachable-tube", required=True)
    ap.add_argument("--development-selected", action="store_true")
    ap.add_argument("--stage", choices=["counterfactual48", "fresh37", "exact200"], default="counterfactual48")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = {
        "cowp": _load(args.cowp), "rvr": _load(args.rvr), "v33_rosh": _load(args.v33_rosh),
        "v35_control_projected_spectrum": _load(args.v35_control_projected_spectrum),
        "v36_recovery_frontier": _load(args.v36_recovery_frontier),
        "v37_recourse_returnability_bridge": _load(args.v37_recourse_returnability_bridge),
        "shift_closed_control_reachable_tube": _load(args.shift_closed_control_reachable_tube),
    }
    data = {k: v for k, v in data.items() if v is not None}
    rowmap = {k: _rows(v) for k, v in data.items()}
    ids = list(rowmap["cowp"].keys()); idset = set(ids)
    mismatch = {k: sorted(idset.symmetric_difference(set(rows))) for k, rows in rowmap.items() if set(rows) != idset}
    if mismatch: raise SystemExit(f"scenario set mismatch: {mismatch}")

    out: dict[str, Any] = {
        "schema_version": "cowp_shift_closed_control_reachable_tube_analysis_v1", "development_selected": bool(args.development_selected),
        "paper_evidence": False, "scenario_count": len(ids), "stage": str(args.stage), "methods": {}, "paired_vs_cowp": {},
    }
    for name, rows in rowmap.items():
        out["methods"][name] = _aggregate(rows, ids)
        if name != "cowp": out["paired_vs_cowp"][name] = _paired(rowmap["cowp"], rows, ids)

    name = "shift_closed_control_reachable_tube"
    if "rvr" in rowmap:
        old_rescued = set(out["paired_vs_cowp"]["rvr"]["Collision"]["rescued_ids"])
        old_induced = set(out["paired_vs_cowp"]["rvr"]["Collision"]["induced_ids"])
        out["rvr_counterexample_retention"] = {
            name: {
                "old_rvr_rescues_total": len(old_rescued),
                "old_rvr_rescues_retained": int(sum(not _event(rowmap[name][s], "CollisionRate") for s in old_rescued)),
                "old_rvr_induced_total": len(old_induced),
                "old_rvr_induced_avoided": int(sum(not _event(rowmap[name][s], "CollisionRate") for s in old_induced)),
            }
        }
    if "v35_control_projected_spectrum" in rowmap:
        out["paired_vs_v35_control_projected_spectrum"] = _paired(rowmap["v35_control_projected_spectrum"], rowmap[name], ids)
    if "v36_recovery_frontier" in rowmap:
        out["paired_vs_v36_recovery_frontier"] = _paired(rowmap["v36_recovery_frontier"], rowmap[name], ids)
    if "v37_recourse_returnability_bridge" in rowmap:
        out["paired_vs_v37_recourse_returnability_bridge"] = _paired(
            rowmap["v37_recourse_returnability_bridge"], rowmap[name], ids
        )
    if "v33_rosh" in rowmap:
        out["paired_vs_v33_rosh"] = _paired(rowmap["v33_rosh"], rowmap[name], ids)

    pc = out["paired_vs_cowp"][name]
    ep_delta = pc["EP"]["delta_mean"]
    switch_rate = out["methods"][name].get("recovery_tube_action_change_step_rate")
    if args.stage == "counterfactual48":
        ret = out["rvr_counterexample_retention"][name]
        checks = {
            "retain_at_least_5_of_10_old_rvr_rescues": ret["old_rvr_rescues_retained"] >= 5,
            "avoid_at_least_7_of_9_old_rvr_induced": ret["old_rvr_induced_avoided"] >= 7,
            "net_remove_at_least_3_cowp_collisions": pc["Collision"]["net_failures_removed"] >= 3,
            "kinematics_net_regression_at_most_1_scene": pc["Kinematics"]["net_failures_removed"] >= -1,
            "mean_ep_delta_not_below_minus_0_05": ep_delta is not None and ep_delta >= -0.05,
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    elif args.stage == "fresh37":
        checks = {
            "no_net_collision_harm": pc["Collision"]["net_failures_removed"] >= 0,
            "no_net_cr_harm": pc["CR"]["net_failures_removed"] >= 0,
            "offroad_net_regression_at_most_1_scene": pc["Offroad"]["net_failures_removed"] >= -1,
            "kinematics_net_regression_at_most_1_scene": pc["Kinematics"]["net_failures_removed"] >= -1,
            "mean_ep_delta_not_below_minus_0_03": ep_delta is not None and ep_delta >= -0.03,
            "nonzero_intervention": switch_rate is not None and switch_rate > 0.0,
        }
    else:
        checks = {"development_confirmation_only": True}
    out["preregistered_gate"] = {name: {"checks": checks, "pass": bool(all(checks.values())), "paper_evidence": False}}
    # Mechanism diagnostics are deliberately NOT new outcome gates.  In
    # particular, a near-zero lifted-selection rate would mean that any outcome
    # gain came from the terminal tube certificate rather than newly constructed
    # control support; this affects the claim, not the frozen six-item gate.
    out["control_reachable_tube_mechanism_diagnostic"] = {
        "tube_probe_step_rate": out["methods"][name].get("recovery_tube_probe_step_rate"),
        "tube_certificate_step_rate": out["methods"][name].get("recovery_tube_certificate_step_rate"),
        "tube_action_change_step_rate": out["methods"][name].get("recovery_tube_action_change_step_rate"),
        "lifted_selection_rate_on_certified_steps": out["methods"][name].get("recovery_tube_lifted_selection_rate_on_certified_steps"),
        "mean_parent_action_classes_on_probes": out["methods"][name].get("mean_recovery_tube_parent_action_classes_on_probes"),
        "mean_generated_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_hypotheses_generated_on_probes"),
        "mean_unique_first_actions_on_probes": out["methods"][name].get("mean_recovery_tube_unique_action_hypotheses_on_probes"),
        "mean_full_safe_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_full_physically_safe_on_probes"),
        "mean_shift_closed_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_shift_closed_on_probes"),
        "mean_nominal_shift_closed_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_nominal_shift_closed_on_probes"),
        "mean_lower_envelope_shift_closed_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_lower_envelope_shift_closed_on_probes"),
        "mean_upper_envelope_shift_closed_tubes_on_probes": out["methods"][name].get("mean_recovery_tube_upper_envelope_shift_closed_on_probes"),
        "mean_lifted_only_parent_count_on_probes": out["methods"][name].get("mean_recovery_tube_lifted_only_parent_count_on_probes"),
        "mean_abs_first_accel_delta_on_certified_steps": out["methods"][name].get("mean_recovery_tube_abs_first_accel_delta_on_certified_steps"),
        "interpretation": (
            "The inherited six-item promotion gate is unchanged. A nonzero lifted-selection/lifted-only-support signal is required only to attribute success specifically to constructive control-support expansion; it is not a post-hoc seventh outcome gate."
        ),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
