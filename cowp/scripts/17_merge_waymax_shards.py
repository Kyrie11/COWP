from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cowp.waymax_eval.metrics_standard import aggregate_waymax_standard_metrics


def _numeric(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _weighted_dict(payloads: list[dict], key: str, weight_key: str) -> dict[str, float]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    for payload in payloads:
        block = payload.get(key, {}) or {}
        default_weight = max(float(payload.get("num_rollouts", 0) or 0), 1.0)
        weight = _numeric(block.get(weight_key)) or default_weight
        for name, value in block.items():
            val = _numeric(value)
            if val is not None:
                buckets.setdefault(name, []).append((val, weight))
    out: dict[str, float] = {}
    for name, vals in buckets.items():
        den = sum(w for _, w in vals)
        if den > 0:
            out[name] = float(sum(v * w for v, w in vals) / den)
    return out


def merge_payloads(paths: list[Path]) -> dict:
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if not payloads:
        raise ValueError("No shard payloads were provided")
    merged = {k: v for k, v in payloads[0].items() if k not in {
        "shard_index", "steps", "standard_metrics", "standard_metric_summary",
        "policy_diagnostic_summary", "closed_loop_cowp_metric_summary",
    }}
    merged["merged_from"] = [str(p) for p in paths]
    merged["num_shards"] = len(paths)
    merged["num_rollouts"] = int(sum(int(p.get("num_rollouts", 0) or 0) for p in payloads))
    merged["steps"] = [int(x) for p in payloads for x in (p.get("steps", []) or [])]
    standard_metrics = [m for p in payloads for m in (p.get("standard_metrics", []) or [])]
    merged["standard_metrics"] = standard_metrics
    merged["standard_metric_summary"] = aggregate_waymax_standard_metrics(
        [{"standard_metrics": m} for m in standard_metrics]
    )
    merged["policy_diagnostic_summary"] = _weighted_dict(
        payloads, "policy_diagnostic_summary", "ClosedLoopPolicySteps"
    )
    merged["closed_loop_cowp_metric_summary"] = _weighted_dict(
        payloads, "closed_loop_cowp_metric_summary", "EpisodesWithDiagnostics"
    )
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge COWP Waymax shard JSONs without losing per-episode metrics.")
    ap.add_argument("--output", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    paths = [Path(x) for x in args.shards]
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing/empty shard files: {missing}")
    merged = merge_payloads(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "num_rollouts": merged["num_rollouts"],
        "standard_metric_summary": merged["standard_metric_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
