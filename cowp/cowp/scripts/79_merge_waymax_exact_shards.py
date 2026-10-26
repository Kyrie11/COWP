from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge disjoint exact-ID Waymax JSON shards without losing scenario pairing.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    payloads = [json.load(open(x, encoding="utf-8")) for x in args.inputs]
    if not payloads:
        raise SystemExit("no inputs")
    method = str(payloads[0].get("method"))
    for p in payloads:
        if str(p.get("method")) != method:
            raise ValueError("all shards must use the same method")
        if p.get("scenario_ids_sha256") != payloads[0].get("scenario_ids_sha256"):
            raise ValueError("scenario manifest hash mismatch")
    records: list[dict[str, Any]] = []
    for p in payloads:
        ids = [str(x) for x in p.get("scenario_ids_resolved", [])]
        metrics = list(p.get("standard_metrics", []))
        diags = list(p.get("scenario_diagnostics", []))
        if metrics and len(metrics) != len(ids):
            raise ValueError("standard_metrics length mismatch")
        if diags and len(diags) != len(ids):
            raise ValueError("scenario_diagnostics length mismatch")
        for i, sid in enumerate(ids):
            records.append({
                "scenario_id": sid,
                "standard_metrics": metrics[i] if i < len(metrics) else {},
                "diagnostics": diags[i] if i < len(diags) else {},
            })
    ids = [r["scenario_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids across shards")
    buckets: dict[str, list[float]] = {}
    for r in records:
        for k, v in (r["standard_metrics"] or {}).items():
            nv = _num(v)
            if nv is not None:
                buckets.setdefault(k, []).append(nv)
    std_summary = {k: float(np.mean(v)) for k, v in sorted(buckets.items()) if v}
    steps = sum(int((r["diagnostics"] or {}).get("steps", 0) or 0) for r in records)
    if steps <= 0:
        steps = sum(sum(int(x) for x in p.get("steps", [])) for p in payloads)
    fb_num = 0.0
    fb_den = 0.0
    for r in records:
        d = r["diagnostics"] or {}
        n = float(d.get("steps", 0) or 0)
        fb_num += n * float(d.get("fallback_step_rate", 0.0) or 0.0)
        fb_den += n
    out = {
        "schema_version": "cowp_waymax_exact_shard_merge_v1",
        "method": method,
        "checkpoint": payloads[0].get("checkpoint"),
        "scenario_ids_sha256": payloads[0].get("scenario_ids_sha256"),
        "num_rollouts": len(records),
        "num_shards_merged": len(payloads),
        "input_files": args.inputs,
        "standard_metric_summary": std_summary,
        "ClosedLoopFallbackStepRate": float(fb_num / fb_den) if fb_den else None,
        "scenario_results": records,
        "shard_policy_diagnostic_summaries": [p.get("policy_diagnostic_summary", {}) for p in payloads],
        "shard_failure_attribution_summaries": [p.get("physical_failure_attribution_summary", {}) for p in payloads],
        "shard_runtime_profiles": [p.get("waymax_runtime_profile_summary", {}) for p in payloads],
    }
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("method", "num_rollouts", "standard_metric_summary", "ClosedLoopFallbackStepRate")}, indent=2))


if __name__ == "__main__":
    main()
