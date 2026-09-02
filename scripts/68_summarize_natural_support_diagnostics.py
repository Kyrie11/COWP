from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bucket_future(valid_steps: int) -> str:
    n = int(valid_steps)
    if n <= 0:
        return "0"
    if n < 20:
        return "1-19"
    if n < 40:
        return "20-39"
    if n < 60:
        return "40-59"
    if n < 80:
        return "60-79"
    return "80"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v16.8.13 pair-neutral/natural-root construction diagnostics with auditability coverage.")
    ap.add_argument("--input", required=True, help="01_build_labels_from_proto --profile-jsonl output")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    profile = Path(args.input)
    rows = 0
    written = 0
    critical_selected = 0
    critical = 0  # mechanism-auditable selected critical agents
    unauditable = 0
    unauditable_future_support: Counter[str] = Counter()
    unauditable_finalizer: Counter[str] = Counter()
    unauditable_priority: Counter[str] = Counter()
    unauditable_reference: Counter[str] = Counter()
    unauditable_sufficient_future = 0
    unauditable_insufficient_future = 0
    protected = 0
    protected_without_prio = 0
    empirical_roots = 0
    empirical_eligible_agents = 0
    short_route_candidate_agents = 0
    rootless_short_route_candidate_agents = 0
    driveway_context_agents = 0
    rootless = 0
    lt2_low = 0
    neutral_unsafe = 0
    obs_ineligible = 0
    ref_kind: Counter[str] = Counter()
    neutral_source: Counter[str] = Counter()
    rejection: Counter[str] = Counter()
    attempted: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    future_total: Counter[str] = Counter()
    future_rootless: Counter[str] = Counter()
    future_lt2: Counter[str] = Counter()
    rootless_priority: Counter[str] = Counter()
    lt2_priority: Counter[str] = Counter()
    rootless_reference: Counter[str] = Counter()
    lt2_reference: Counter[str] = Counter()
    rootless_dominant_rejection: Counter[str] = Counter()
    lt2_dominant_rejection: Counter[str] = Counter()
    priority_rejection_reason: Counter[str] = Counter()
    rootless_priority_rejection_reason: Counter[str] = Counter()
    lt2_priority_rejection_reason: Counter[str] = Counter()
    map_rejected_min_distances: list[float] = []
    map_rejected_max_distances: list[float] = []
    best_rejected_burdens: list[float] = []
    examples = defaultdict(list)
    parse_errors = []

    if profile.is_file():
        for lineno, line in enumerate(profile.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                if len(parse_errors) < 20:
                    parse_errors.append({"line": lineno, "error": repr(exc)})
                continue
            if row.get("status") != "written":
                continue
            written += 1
            sid = str(row.get("scenario_id", ""))
            eng = row.get("engine_diagnostics", {}) or {}
            neutrals = eng.get("pair_neutral", []) or []
            naturals = eng.get("natural", []) or []
            neutral_by_slot = {int(x.get("slot", -1)): x for x in neutrals if isinstance(x, dict)}
            for d in naturals:
                if not isinstance(d, dict) or not bool(d.get("valid", False)):
                    continue
                critical_selected += 1
                slot = int(d.get("slot", -1))
                steps = int(d.get("future_valid_steps", 0))
                bucket = _bucket_future(steps)
                mechanism_valid = bool(d.get("mechanism_valid", True))
                if not mechanism_valid:
                    unauditable += 1
                    unauditable_future_support[bucket] += 1
                    finalizer = str(d.get("auditability_finalizer", "precheck_insufficient_evidence"))
                    unauditable_finalizer[finalizer] += 1
                    unauditable_priority[str(d.get("rho", "unknown"))] += 1
                    unauditable_reference[str(d.get("reference_kind", "unauditable"))] += 1
                    frac = float(d.get("future_valid_fraction", 0.0) or 0.0)
                    if steps >= 60 and frac >= 0.70:
                        # With the current natural builder this means the actor
                        # had enough valid factual timestamps but neither a full
                        # routable lane continuation nor non-degenerate factual
                        # path segments.  Treat this as an explicit unknown
                        # mechanism target rather than manufacturing option roots.
                        unauditable_sufficient_future += 1
                    else:
                        unauditable_insufficient_future += 1
                    if len(examples["unauditable"]) < 50:
                        examples["unauditable"].append({
                            "scenario_id": sid, "slot": slot, "track_index": d.get("track_index"),
                            "rho": d.get("rho"), "future_valid_steps": steps,
                            "future_valid_fraction": d.get("future_valid_fraction"),
                            "reference_kind": d.get("reference_kind", "unauditable"),
                            "auditability_finalizer": d.get("auditability_finalizer", "precheck_insufficient_evidence"),
                        })
                    continue
                critical += 1
                future_total[bucket] += 1
                roots = int(d.get("root_count", 0))
                low = int(d.get("low_burden_root_count", 0))
                rho_int = int(d.get("rho", -1))
                prio_roots = int(d.get("prio_root_count", 0))
                empirical_roots += int(d.get("empirical_corridor_root_count", 0))
                empirical_eligible_agents += int(bool(d.get("empirical_corridor_eligible", False)))
                route_poly_count = int(d.get("route_polyline_count", 0))
                full_route_count = int(d.get("full_horizon_map_route_count", 0))
                short_route_candidate = route_poly_count > 0 and full_route_count == 0
                short_route_candidate_agents += int(short_route_candidate)
                driveway_context_agents += int(int(d.get("driveway_polygon_count", 0)) > 0)
                if rho_int in (2, 3):
                    protected += 1
                    if prio_roots <= 0:
                        protected_without_prio += 1
                        if len(examples["protected_without_prio"]) < 50:
                            examples["protected_without_prio"].append({
                                "scenario_id": sid, "slot": slot, "track_index": d.get("track_index"),
                                "rho": rho_int, "root_count": roots, "low_burden_root_count": low,
                                "reference_kind": d.get("reference_kind"), "accepted_by_phase": d.get("accepted_by_phase", {}),
                            })
                is_rootless = roots <= 0
                is_lt2 = low < 2
                reject_row = {str(k): int(v) for k, v in (d.get("rejection_counts", {}) or {}).items()}
                dominant = max(reject_row.items(), key=lambda kv: (kv[1], kv[0]))[0] if reject_row and max(reject_row.values()) > 0 else "none"
                rho = str(d.get("rho", "unknown"))
                ref = str(d.get("reference_kind", "unknown"))
                pr_reasons = {str(k): int(v) for k, v in (d.get("priority_rejection_reasons", {}) or {}).items()}
                for k, v in pr_reasons.items():
                    priority_rejection_reason[k] += v
                min_map_rej = d.get("map_rejected_min_max_distance_m")
                max_map_rej = d.get("map_rejected_max_max_distance_m")
                best_burden = d.get("best_rejected_burden")
                if min_map_rej is not None:
                    try: map_rejected_min_distances.append(float(min_map_rej))
                    except Exception: pass
                if max_map_rej is not None:
                    try: map_rejected_max_distances.append(float(max_map_rej))
                    except Exception: pass
                if best_burden is not None:
                    try: best_rejected_burdens.append(float(best_burden))
                    except Exception: pass
                if is_rootless:
                    rootless += 1
                    rootless_short_route_candidate_agents += int(short_route_candidate)
                    future_rootless[bucket] += 1
                    rootless_priority[rho] += 1
                    rootless_reference[ref] += 1
                    rootless_dominant_rejection[dominant] += 1
                    for k, v in pr_reasons.items():
                        rootless_priority_rejection_reason[k] += v
                    if len(examples["rootless"]) < 50:
                        examples["rootless"].append({
                            "scenario_id": sid, "slot": slot, "track_index": d.get("track_index"),
                            "rho": d.get("rho"), "future_valid_steps": steps, "reference_kind": ref,
                            "route_polyline_count": route_poly_count, "full_horizon_map_route_count": full_route_count,
                            "empirical_corridor_eligible": d.get("empirical_corridor_eligible"),
                            "driveway_polygon_count": d.get("driveway_polygon_count"),
                            "rejection_counts": reject_row, "dominant_rejection": dominant,
                            "priority_rejection_reasons": pr_reasons,
                            "map_rejected_min_max_distance_m": min_map_rej,
                            "map_rejected_max_max_distance_m": max_map_rej,
                            "best_rejected_burden": best_burden,
                        })
                if is_lt2:
                    lt2_low += 1
                    future_lt2[bucket] += 1
                    lt2_priority[rho] += 1
                    lt2_reference[ref] += 1
                    lt2_dominant_rejection[dominant] += 1
                    for k, v in pr_reasons.items():
                        lt2_priority_rejection_reason[k] += v
                    if len(examples["lt2_low"]) < 50:
                        examples["lt2_low"].append({
                            "scenario_id": sid, "slot": slot, "track_index": d.get("track_index"),
                            "rho": d.get("rho"), "root_count": roots, "low_burden_root_count": low,
                            "future_valid_steps": steps, "reference_kind": ref, "rejection_counts": reject_row,
                            "dominant_rejection": dominant, "priority_rejection_reasons": pr_reasons,
                            "map_rejected_min_max_distance_m": min_map_rej,
                            "map_rejected_max_max_distance_m": max_map_rej,
                            "best_rejected_burden": best_burden,
                        })
                if not bool(d.get("obs_eligible", False)):
                    obs_ineligible += 1
                ref_kind[str(d.get("reference_kind", "unknown"))] += 1
                for k, v in (d.get("rejection_counts", {}) or {}).items():
                    rejection[str(k)] += int(v)
                for k, v in (d.get("attempted_by_phase", {}) or {}).items():
                    attempted[str(k)] += int(v)
                for k, v in (d.get("accepted_by_phase", {}) or {}).items():
                    accepted[str(k)] += int(v)

                nd = neutral_by_slot.get(slot, {})
                if nd:
                    neutral_source[str(nd.get("neutral_source", "unknown"))] += 1
                    if bool(nd.get("neutral_actor_unsafe", False)):
                        neutral_unsafe += 1
                        if len(examples["neutral_unsafe"]) < 50:
                            examples["neutral_unsafe"].append({"scenario_id": sid, "slot": slot, "track_index": d.get("track_index"), "neutral_source": nd.get("neutral_source"), "neutral_actor_burden": nd.get("neutral_actor_burden"), "neutral_min_distance_m": nd.get("neutral_min_distance_m")})

    def rate(n: int, d: int) -> float:
        return float(n / max(d, 1))

    def quantiles(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "p50": None, "p90": None, "max": None}
        vals = sorted(values)
        def q(frac: float) -> float:
            pos = int(round(frac * (len(vals) - 1)))
            return float(vals[max(0, min(pos, len(vals) - 1))])
        return {"min": float(vals[0]), "p50": q(0.50), "p90": q(0.90), "max": float(vals[-1])}

    report = {
        "schema_version": "cowp_v16_8_17_natural_support_diagnostic_v5",
        "profile_jsonl": str(profile.resolve()),
        "rows": rows,
        "written_scenes": written,
        "critical_agents_selected": critical_selected,
        "critical_agents": critical,
        "mechanism_auditable_critical_agents": critical,
        "mechanism_unauditable_critical_agents": unauditable,
        "mechanism_unauditable_rate": rate(unauditable, critical_selected),
        "mechanism_unauditable_future_support": dict(unauditable_future_support),
        "mechanism_unauditable_finalizer_counts": dict(unauditable_finalizer),
        "mechanism_unauditable_by_priority_relation": dict(unauditable_priority),
        "mechanism_unauditable_by_reference_kind": dict(unauditable_reference),
        "mechanism_unauditable_with_sufficient_future_but_no_substantial_route_geometry": unauditable_sufficient_future,
        "mechanism_unauditable_with_insufficient_future": unauditable_insufficient_future,
        "protected_auditable_critical_agents": protected,
        "protected_without_prio_root": protected_without_prio,
        "protected_prio_root_coverage": float((protected - protected_without_prio) / max(protected, 1)) if protected else 1.0,
        "empirical_corridor_roots": empirical_roots,
        "empirical_corridor_eligible_agents": empirical_eligible_agents,
        "short_route_candidate_agents": short_route_candidate_agents,
        "rootless_short_route_candidate_agents": rootless_short_route_candidate_agents,
        "driveway_context_agents": driveway_context_agents,
        "rootless_critical_agents": rootless,
        "rootless_rate": rate(rootless, critical),
        "critical_agents_with_lt2_low_burden_roots": lt2_low,
        "lt2_low_burden_rate": rate(lt2_low, critical),
        "pair_neutral_unsafe_agents": neutral_unsafe,
        "pair_neutral_unsafe_rate": rate(neutral_unsafe, critical),
        "obs_ineligible_agents": obs_ineligible,
        "obs_ineligible_rate": rate(obs_ineligible, critical),
        "reference_kind_counts": dict(ref_kind),
        "pair_neutral_source_counts": dict(neutral_source),
        "natural_rejection_counts": dict(rejection),
        "attempted_by_phase": dict(attempted),
        "accepted_by_phase": dict(accepted),
        "rootless_by_priority_relation": dict(rootless_priority),
        "lt2_low_burden_by_priority_relation": dict(lt2_priority),
        "rootless_by_reference_kind": dict(rootless_reference),
        "lt2_low_burden_by_reference_kind": dict(lt2_reference),
        "rootless_dominant_rejection": dict(rootless_dominant_rejection),
        "lt2_low_burden_dominant_rejection": dict(lt2_dominant_rejection),
        "priority_rejection_reason_counts": dict(priority_rejection_reason),
        "rootless_priority_rejection_reason_counts": dict(rootless_priority_rejection_reason),
        "lt2_low_burden_priority_rejection_reason_counts": dict(lt2_priority_rejection_reason),
        "map_rejected_min_distance_summary_m": quantiles(map_rejected_min_distances),
        "map_rejected_max_distance_summary_m": quantiles(map_rejected_max_distances),
        "best_rejected_burden_summary": quantiles(best_rejected_burdens),
        "future_support": {
            b: {
                "critical_agents": int(future_total[b]),
                "rootless": int(future_rootless[b]),
                "rootless_rate": rate(int(future_rootless[b]), int(future_total[b])),
                "lt2_low_burden": int(future_lt2[b]),
                "lt2_low_burden_rate": rate(int(future_lt2[b]), int(future_total[b])),
            }
            for b in ("0", "1-19", "20-39", "40-59", "60-79", "80")
            if future_total[b]
        },
        "examples": dict(examples),
        "parse_errors": parse_errors,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("critical_agents_selected", "mechanism_auditable_critical_agents", "mechanism_unauditable_critical_agents", "mechanism_unauditable_rate", "rootless_critical_agents", "rootless_rate", "critical_agents_with_lt2_low_burden_roots", "lt2_low_burden_rate", "protected_without_prio_root", "empirical_corridor_roots")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
