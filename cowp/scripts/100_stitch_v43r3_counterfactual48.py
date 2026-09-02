from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch disjoint gate19 + remaining29 V43R3 results into the exact frozen counterfactual48 merged result.")
    ap.add_argument("--gate19", required=True)
    ap.add_argument("--remaining29", required=True)
    ap.add_argument("--full-ids", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    parts = [_load(args.gate19), _load(args.remaining29)]
    method = str(parts[0].get("method"))
    checkpoint = parts[0].get("checkpoint")
    for p in parts:
        if str(p.get("method")) != method:
            raise SystemExit("method mismatch across gate19/remaining29")
        if p.get("checkpoint") != checkpoint:
            raise SystemExit("checkpoint mismatch across gate19/remaining29")

    rows: dict[str, dict[str, Any]] = {}
    for p in parts:
        for row in p.get("scenario_results", []):
            sid = str(row["scenario_id"])
            if sid in rows:
                raise SystemExit(f"duplicate scenario across subsets: {sid}")
            rows[sid] = row

    full_ids = [x.strip() for x in Path(args.full_ids).read_text().splitlines() if x.strip()]
    if len(full_ids) != 48 or len(set(full_ids)) != 48:
        raise SystemExit("full counterfactual manifest must contain 48 unique IDs")
    if set(rows) != set(full_ids):
        missing = sorted(set(full_ids) - set(rows)); extra = sorted(set(rows) - set(full_ids))
        raise SystemExit(f"stitched ID mismatch missing={missing} extra={extra}")
    ordered = [rows[sid] for sid in full_ids]

    buckets: dict[str, list[float]] = {}
    fb_num = fb_den = 0.0
    for row in ordered:
        for k, v in (row.get("standard_metrics", {}) or {}).items():
            nv = _num(v)
            if nv is not None:
                buckets.setdefault(k, []).append(nv)
        dg = row.get("diagnostics", {}) or {}
        n = float(dg.get("steps", 0) or 0)
        fb_num += n * float(dg.get("fallback_step_rate", 0.0) or 0.0)
        fb_den += n

    full_hash = hashlib.sha256("\n".join(full_ids).encode()).hexdigest()
    out = {
        "schema_version": "cowp_waymax_exact_stitched_subset_merge_v1",
        "method": method,
        "checkpoint": checkpoint,
        "scenario_ids_sha256": full_hash,
        "num_rollouts": 48,
        "num_shards_merged": sum(int(p.get("num_shards_merged", 0) or 0) for p in parts),
        "input_files": [args.gate19, args.remaining29],
        "standard_metric_summary": {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v},
        "ClosedLoopFallbackStepRate": (fb_num / fb_den) if fb_den else None,
        "scenario_results": ordered,
        "shard_policy_diagnostic_summaries": [x for p in parts for x in p.get("shard_policy_diagnostic_summaries", [])],
        "shard_failure_attribution_summaries": [x for p in parts for x in p.get("shard_failure_attribution_summaries", [])],
        "shard_runtime_profiles": [x for p in parts for x in p.get("shard_runtime_profiles", [])],
        "stitch_provenance": {
            "gate19_n": len(parts[0].get("scenario_results", [])),
            "remaining29_n": len(parts[1].get("scenario_results", [])),
            "full_manifest": args.full_ids,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": method, "n": 48, "scenario_ids_sha256": full_hash, "standard_metric_summary": out["standard_metric_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
