from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def _rows(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["scenario_id"]): r for r in d.get("scenario_results", [])}


def _metric(row: dict[str, Any], key: str) -> float:
    v = (row.get("standard_metrics", {}) or {}).get(key)
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return float("nan")
    return float(v)


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


def _bootstrap_mean_ci(x: np.ndarray, seed: int = 16835, draws: int = 10000) -> list[float] | None:
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
    diag_keys = [
        "fallback_step_rate",
        "zero_conventional_candidate_step_rate",
        "mean_conventional_candidates",
        "mean_max_collision_safe_prefix_steps",
        "mean_selected_collision_safe_prefix_steps",
        "recovery_switch_step_rate",
        "successor_option_probe_step_rate",
        "second_successor_option_probe_step_rate",
        "mean_successor_signature_compare_on_probes",
        "mean_second_successor_signature_compare_on_probes",
        "recovery_commitment_active_step_rate",
        "recovery_commitment_entry_step_rate",
        "recovery_commitment_continue_step_rate",
        "recovery_commitment_clear_step_rate",
        "mean_recovery_prefix_gain_steps",
        "mean_recovery_action_risk_delta",
        "mean_recovery_rule_risk_delta",
        "mean_recovery_pressure_risk_delta",
        "recovery_hysteresis_active_step_rate",
        "recovery_hysteresis_entry_step_rate",
        "recovery_hysteresis_continue_step_rate",
        "recovery_hysteresis_exit_step_rate",
        "recovery_hysteresis_clear_step_rate",
        "recovery_option_profile_probe_step_rate",
        "recovery_option_profile_strict_dominance_rate_on_probes",
        "recovery_option_profile_weak_dominance_rate_on_probes",
        "mean_recovery_option_profile_min_margin_on_probes",
        "mean_recovery_option_profile_area_delta_on_probes",
        "recovery_executable_option_profile_probe_step_rate",
        "recovery_base_controller_transition_feasible_rate_on_probes",
        "recovery_rvr_controller_transition_feasible_rate_on_probes",
        "mean_recovery_controller_transition_delta_on_probes",
        "mean_recovery_base_transition_feasible_candidates_on_exec_probes",
        "mean_recovery_rvr_transition_feasible_candidates_on_exec_probes",
        "mean_recovery_base_transition_rejected_roadgraph_candidates_on_exec_probes",
        "mean_recovery_rvr_transition_rejected_roadgraph_candidates_on_exec_probes",
        "selected_controller_transition_feasible_step_rate",
        "transition_guarded_rosh_step_rate",
        "executable_option_spectrum_hysteresis_step_rate",
        "waymax_kinematic_guarded_rosh_step_rate",
        "control_projected_option_spectrum_hysteresis_step_rate",
        "recovery_waymax_kinematic_guard_probe_step_rate",
        "recovery_control_projected_option_profile_probe_step_rate",
        "recovery_base_waymax_kinematic_feasible_rate_on_probes",
        "recovery_rvr_waymax_kinematic_feasible_rate_on_probes",
        "mean_recovery_waymax_kinematic_transition_delta_on_probes",
        "mean_recovery_base_waymax_abs_steering_curvature_on_probes",
        "mean_recovery_rvr_waymax_abs_steering_curvature_on_probes",
        "mean_recovery_base_control_projected_h1_kinematic_feasible_candidates",
        "mean_recovery_rvr_control_projected_h1_kinematic_feasible_candidates",
        "mean_recovery_base_control_projected_full_kinematic_feasible_candidates",
        "mean_recovery_rvr_control_projected_full_kinematic_feasible_candidates",
        "selected_waymax_kinematic_feasible_step_rate",
        "selected_waymax_kinematic_feasible_rate_on_recovery_switch_steps",
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
    event_keys = {
        "CR": "CR",
        "Collision": "CollisionRate",
        "Offroad": "OffroadRate",
        "Kinematics": "KinematicsInfeasibilityRate",
    }
    for label, key in event_keys.items():
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
    dep = []
    for s in ids:
        a = _metric(base[s], "EP")
        b = _metric(alt[s], "EP")
        if math.isfinite(a) and math.isfinite(b):
            dep.append(b - a)
    arr = np.asarray(dep, dtype=np.float64)
    out["EP"] = {
        "paired_n": int(arr.size),
        "delta_mean": float(arr.mean()) if arr.size else None,
        "delta_median": float(np.median(arr)) if arr.size else None,
        "bootstrap95": _bootstrap_mean_ci(arr),
    }
    return out


def _hybrid_only(cowp: dict[str, dict[str, Any]], rvr: dict[str, dict[str, Any]], alt: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    """Find event changes that neither pure endpoint policy exhibits.

    These cases are evidence that switching between COWP and RVR creates a third
    closed-loop regime rather than merely interpolating their outcomes.
    """
    out: dict[str, Any] = {}
    for label, key in {"Collision": "CollisionRate", "Offroad": "OffroadRate", "Kinematics": "KinematicsInfeasibilityRate"}.items():
        induced = [s for s in ids if not _event(cowp[s], key) and not _event(rvr[s], key) and _event(alt[s], key)]
        rescued = [s for s in ids if _event(cowp[s], key) and _event(rvr[s], key) and not _event(alt[s], key)]
        out[label] = {"hybrid_only_induced": induced, "hybrid_only_rescued": rescued}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--rvr")
    ap.add_argument("--v33-rosh")
    ap.add_argument("--v34-transition-guarded")
    ap.add_argument("--v34-executable-spectrum")
    ap.add_argument("--waymax-kinematic-guarded")
    ap.add_argument("--control-projected-spectrum")
    ap.add_argument("--development-selected", action="store_true")
    ap.add_argument("--stage", choices=["counterfactual48", "fresh37", "exact200"], default="counterfactual48")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = {
        "cowp": _load(args.cowp),
        "rvr": _load(args.rvr),
        "v33_rosh": _load(args.v33_rosh),
        "v34_transition_guarded": _load(args.v34_transition_guarded),
        "v34_executable_spectrum": _load(args.v34_executable_spectrum),
        "waymax_kinematic_guarded": _load(args.waymax_kinematic_guarded),
        "control_projected_spectrum": _load(args.control_projected_spectrum),
    }
    data = {k: v for k, v in data.items() if v is not None}
    rowmap = {k: _rows(v) for k, v in data.items()}
    ids = list(rowmap["cowp"].keys())
    idset = set(ids)
    mismatch = {k: sorted(idset.symmetric_difference(set(rows))) for k, rows in rowmap.items() if set(rows) != idset}
    if mismatch:
        raise SystemExit(f"scenario set mismatch: {mismatch}")

    out: dict[str, Any] = {
        "schema_version": "cowp_control_projected_option_spectrum_analysis_v1",
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

    candidates = [n for n in ("waymax_kinematic_guarded", "control_projected_spectrum") if n in rowmap]

    if "rvr" in rowmap:
        old_rescued = set(out["paired_vs_cowp"]["rvr"]["Collision"]["rescued_ids"])
        old_induced = set(out["paired_vs_cowp"]["rvr"]["Collision"]["induced_ids"])
        out["rvr_counterexample_retention"] = {}
        for name in [n for n in ("v33_rosh", "v34_transition_guarded", "v34_executable_spectrum", *candidates) if n in rowmap]:
            rows = rowmap[name]
            out["rvr_counterexample_retention"][name] = {
                "old_rvr_rescues_total": len(old_rescued),
                "old_rvr_rescues_retained": int(sum(not _event(rows[s], "CollisionRate") for s in old_rescued)),
                "old_rvr_induced_total": len(old_induced),
                "old_rvr_induced_avoided": int(sum(not _event(rows[s], "CollisionRate") for s in old_induced)),
            }
            out.setdefault("hybrid_only_vs_pure_endpoints", {})[name] = _hybrid_only(
                rowmap["cowp"], rowmap["rvr"], rows, ids
            )

    if "v33_rosh" in rowmap:
        out["paired_vs_v33_rosh"] = {name: _paired(rowmap["v33_rosh"], rowmap[name], ids) for name in candidates}
        # Diagnostic only: v16.8.33 failed promotion at the kinematics bound.
        # These known counterexamples must never become an outcome-tuned hard gate.
        old_kin_induced = set(out["paired_vs_cowp"]["v33_rosh"]["Kinematics"]["induced_ids"])
        old_col_rescued = set(out["paired_vs_cowp"]["v33_rosh"]["Collision"]["rescued_ids"])
        old_col_induced = set(out["paired_vs_cowp"]["v33_rosh"]["Collision"]["induced_ids"])
        out["v33_rosh_counterexample_diagnostics"] = {}
        for name in candidates:
            rows = rowmap[name]
            out["v33_rosh_counterexample_diagnostics"][name] = {
                "v33_kinematics_induced_total": len(old_kin_induced),
                "v33_kinematics_induced_avoided": int(sum(not _event(rows[s], "KinematicsInfeasibilityRate") for s in old_kin_induced)),
                "v33_collision_rescues_total": len(old_col_rescued),
                "v33_collision_rescues_retained": int(sum(not _event(rows[s], "CollisionRate") for s in old_col_rescued)),
                "v33_collision_induced_total": len(old_col_induced),
                "v33_collision_induced_avoided": int(sum(not _event(rows[s], "CollisionRate") for s in old_col_induced)),
                "v33_kinematics_induced_ids": sorted(old_kin_induced),
            }

    # V16.8.34 references are included only for attribution: did the new
    # evaluator-aligned predicate recover the recall destroyed by nominal exact-
    # waypoint filtering?  They are not promotion baselines.
    for ref_name in ("v34_transition_guarded", "v34_executable_spectrum"):
        if ref_name in rowmap:
            out[f"paired_vs_{ref_name}"] = {name: _paired(rowmap[ref_name], rowmap[name], ids) for name in candidates}

    if len(candidates) == 2:
        out["paired_control_projected_vs_waymax_guarded"] = _paired(
            rowmap["waymax_kinematic_guarded"], rowmap["control_projected_spectrum"], ids
        )

    # Keep the exact v16.8.33/v16.8.34 promotion gate unchanged.  This avoids moving the
    # threshold after seeing the near-miss: v33 ROSH failed at +2 kinematics while
    # the preregistered contract allowed at most +1.
    out["preregistered_gate"] = {}
    for name in candidates:
        pc = out["paired_vs_cowp"][name]
        ep_delta = pc["EP"]["delta_mean"]
        switch_rate = out["methods"][name].get("recovery_switch_step_rate")
        if args.stage == "counterfactual48" and "rvr_counterexample_retention" in out:
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
        out["preregistered_gate"][name] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "paper_evidence": False,
        }

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
