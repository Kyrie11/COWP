from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize resumable COWP label-build profile JSONL and identify runtime bottlenecks.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-slow", type=int, default=20)
    args = ap.parse_args()

    latest: dict[str, dict[str, Any]] = {}
    malformed = 0
    for raw in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:
            malformed += 1
            continue
        sid = str(row.get("scenario_id", "")).strip()
        if sid:
            latest[sid] = row

    status = Counter(str(r.get("status", "unknown")) for r in latest.values())
    filters = Counter(str(r.get("filter_reason", "filtered")) for r in latest.values() if str(r.get("status")) == "filtered")
    rej = Counter()
    attempted = Counter()
    accepted = Counter()
    critical_modes = Counter()
    critical_counts: list[int] = []
    audit_valid_pairs = 0
    audit_relevant_pairs = 0
    audit_relevance_mass_sum = 0.0
    audit_scenes = 0
    timings: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    slow: list[tuple[float, str, str, dict[str, float]]] = []
    for sid, row in latest.items():
        total = float(row.get("seconds", 0.0) or 0.0)
        totals.append(total)
        tdict = row.get("timings") or {}
        numeric_t: dict[str, float] = {}
        if isinstance(tdict, dict):
            for k, v in tdict.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    timings[str(k)].append(fv)
                    numeric_t[str(k)] = fv
        diag = row.get("candidate_diagnostics") or {}
        engine_diag = row.get("engine_diagnostics") or {}
        if isinstance(engine_diag, dict) and isinstance(engine_diag.get("candidate"), dict):
            diag = engine_diag.get("candidate") or diag
        if isinstance(diag, dict):
            for k, v in (diag.get("rejection_counts") or {}).items():
                try:
                    rej[str(k)] += int(v)
                except Exception:
                    pass
            for k, v in (diag.get("attempted_by_source") or {}).items():
                try:
                    attempted[str(k)] += int(v)
                except Exception:
                    pass
            for k, v in (diag.get("accepted_by_source") or {}).items():
                try:
                    accepted[str(k)] += int(v)
                except Exception:
                    pass
        if isinstance(engine_diag, dict) and isinstance(engine_diag.get("critical"), dict):
            cdiag = engine_diag["critical"]
            critical_modes[str(cdiag.get("selection_reference_mode", "unknown"))] += 1
            try:
                critical_counts.append(int(cdiag.get("count", 0)))
            except Exception:
                pass
        if isinstance(engine_diag, dict) and isinstance(engine_diag.get("audit"), dict):
            adiag = engine_diag["audit"]
            try:
                vp = int(adiag.get("valid_pair_count", 0))
                rp = int(adiag.get("relevant_pair_count", 0))
                audit_valid_pairs += vp
                audit_relevant_pairs += rp
                audit_relevance_mass_sum += float(adiag.get("mean_relevance_mass", 0.0))
                audit_scenes += 1
            except Exception:
                pass
        slow.append((total, sid, str(row.get("status", "unknown")), numeric_t))

    stage_rows = []
    total_stage_sum = sum(sum(v) for v in timings.values())
    for key, vals in timings.items():
        stage_rows.append({
            "stage": key,
            "count": len(vals),
            "mean_s": float(np.mean(vals)) if vals else 0.0,
            "p50_s": _pct(vals, 50),
            "p90_s": _pct(vals, 90),
            "p99_s": _pct(vals, 99),
            "aggregate_s": float(sum(vals)),
            "aggregate_share_of_recorded_stage_time": float(sum(vals) / max(total_stage_sum, 1e-9)),
        })
    stage_rows.sort(key=lambda x: x["aggregate_s"], reverse=True)
    slow.sort(reverse=True)

    result = {
        "schema_version": "cowp_v16_8_9_label_build_profile_summary_v3",
        "input": str(Path(args.input).resolve()),
        "unique_scenarios": len(latest),
        "malformed_rows": malformed,
        "status_counts": dict(status),
        "filter_reason_counts": dict(filters),
        "candidate_rejection_counts_all_profiled_scenes": dict(rej),
        "candidate_attempted_by_source_all_profiled_scenes": dict(attempted),
        "candidate_accepted_by_source_all_profiled_scenes": dict(accepted),
        "candidate_acceptance_rate_by_source": {
            k: float(accepted.get(k, 0) / max(v, 1)) for k, v in attempted.items()
        },
        "critical_selection_reference_modes": dict(critical_modes),
        "audit_relevance": {
            "profiled_scenes": audit_scenes,
            "valid_pair_count": audit_valid_pairs,
            "relevant_pair_count": audit_relevant_pairs,
            "relevant_pair_rate": float(audit_relevant_pairs / max(audit_valid_pairs, 1)),
            "mean_scene_mean_relevance_mass": float(audit_relevance_mass_sum / max(audit_scenes, 1)),
        },
        "critical_agent_count": {
            "mean": float(np.mean(critical_counts)) if critical_counts else 0.0,
            "p50": _pct([float(x) for x in critical_counts], 50),
            "p90": _pct([float(x) for x in critical_counts], 90),
            "max": max(critical_counts) if critical_counts else 0,
        },
        # Backward-compatible aliases used by the v16.8.5 report.
        "zero_valid_candidate_rejection_counts": dict(rej) if status.get("filtered", 0) else {},
        "zero_valid_attempted_by_source": dict(attempted) if status.get("filtered", 0) else {},
        "total_seconds": {
            "mean": float(np.mean(totals)) if totals else 0.0,
            "p50": _pct(totals, 50),
            "p90": _pct(totals, 90),
            "p99": _pct(totals, 99),
            "max": max(totals) if totals else 0.0,
        },
        "stage_breakdown": stage_rows,
        "slowest_scenarios": [
            {"scenario_id": sid, "status": st, "seconds": sec, "timings": t}
            for sec, sid, st, t in slow[: max(0, int(args.top_slow))]
        ],
        "interpretation": (
            "Use stage_breakdown aggregate_share to optimize the dominant label-engine stage. "
            "Do not increase worker count blindly if p90/p99 times or memory pressure are dominated by one CPU-heavy stage."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
