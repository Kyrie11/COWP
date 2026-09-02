from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _eq(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and math.isnan(a):
            return isinstance(b, float) and math.isnan(b)
        if isinstance(b, float) and math.isnan(b):
            return False
        return abs(float(a) - float(b)) <= tol
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify V16.8.43R2 runtime repair preserves profile8 behavioral outcomes.")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference-wall-seconds")
    ap.add_argument("--candidate-wall-seconds")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    ref = _load(args.reference)
    cand = _load(args.candidate)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    add("method_identity", ref.get("method") == cand.get("method"), [ref.get("method"), cand.get("method")])
    add("scenario_hash", ref.get("scenario_ids_sha256") == cand.get("scenario_ids_sha256"), [ref.get("scenario_ids_sha256"), cand.get("scenario_ids_sha256")])
    add("num_rollouts", ref.get("num_rollouts") == cand.get("num_rollouts"), [ref.get("num_rollouts"), cand.get("num_rollouts")])

    rr = {str(x["scenario_id"]): x for x in ref.get("scenario_results", [])}
    cr = {str(x["scenario_id"]): x for x in cand.get("scenario_results", [])}
    add("scenario_id_set", set(rr) == set(cr), {"reference": sorted(rr), "candidate": sorted(cr)})

    mismatches: list[dict[str, Any]] = []
    behavioral_diag_exclusions = (
        "blocker_query",  # intentionally changed by exact-blocker pruning
    )
    for sid in sorted(set(rr) & set(cr)):
        rrow, crow = rr[sid], cr[sid]
        for k in sorted(set(rrow.get("standard_metrics", {})) | set(crow.get("standard_metrics", {}))):
            a = rrow.get("standard_metrics", {}).get(k)
            b = crow.get("standard_metrics", {}).get(k)
            if not _eq(a, b, args.tol):
                mismatches.append({"scenario_id": sid, "field": f"standard_metrics.{k}", "reference": a, "candidate": b})
        rd, cd = rrow.get("diagnostics", {}), crow.get("diagnostics", {})
        for k in sorted(set(rd) & set(cd)):
            if any(token in k for token in behavioral_diag_exclusions):
                continue
            # Runtime repair adds accounting fields and may change cache-hit counts
            # without changing the hard predicate results.  Compare all behavioral
            # diagnostics but exclude cache accounting.
            if "cache_hits" in k:
                continue
            a, b = rd.get(k), cd.get(k)
            if not _eq(a, b, args.tol):
                mismatches.append({"scenario_id": sid, "field": f"diagnostics.{k}", "reference": a, "candidate": b})
    add("behavioral_profile_equivalence", len(mismatches) == 0, {"mismatch_count": len(mismatches), "first_mismatches": mismatches[:50]})

    wall = {}
    if args.reference_wall_seconds and args.candidate_wall_seconds:
        rsec = float(Path(args.reference_wall_seconds).read_text().strip())
        csec = float(Path(args.candidate_wall_seconds).read_text().strip())
        wall = {
            "reference_seconds": rsec,
            "candidate_seconds": csec,
            "speedup": (rsec / csec) if csec > 0 else None,
            "reduction_fraction": ((rsec - csec) / rsec) if rsec > 0 else None,
        }

    out = {
        "schema_version": "v16.8.43r2_runtime_fidelity_v1",
        "reference": args.reference,
        "candidate": args.candidate,
        "tolerance": args.tol,
        "checks": checks,
        "pass": all(x["pass"] for x in checks),
        "wall_clock": wall,
        "interpretation": (
            "PASS means the engineering-only exact-blocker/hypothesis pruning preserved the available profile8 behavioral evidence. "
            "Query/cache accounting is intentionally excluded because those are the optimized work counters."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
