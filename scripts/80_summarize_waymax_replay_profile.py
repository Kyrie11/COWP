from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"count": 0}
    n = len(vals)
    def q(frac: float) -> float:
        if n == 1:
            return vals[0]
        pos = frac * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        w = pos - lo
        return vals[lo] * (1.0 - w) + vals[hi] * w
    return {
        "count": n,
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(q(0.90)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    print(f"[warn] {path}:{line_no}: invalid JSON: {exc}")
                    continue
                if isinstance(row, dict):
                    row = dict(row)
                    row["_profile_file"] = str(path)
                    rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize COWP Waymax replay scene/probe timing JSONL files.")
    ap.add_argument("profiles", nargs="+", help="One or more *_profile_*.jsonl files")
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.profiles]
    rows = _load_rows(paths)
    scene_keys = (
        "load_select_s",
        "load_npz_s",
        "build_state_s",
        "select_candidates_s",
        "env_init_s",
        "rollout_candidates_s",
        "mean_rollout_candidate_s",
        "write_outcomes_s",
        "seconds",
    )
    scene_values: dict[str, list[float]] = defaultdict(list)
    probe_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        status_counts[str(row.get("status", "unknown"))] += 1
        for key in scene_keys:
            v = _finite_float(row.get(key))
            if v is not None:
                scene_values[key].append(v)
        records = row.get("timing_probe_candidates") or []
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict):
                    probe_by_mode[str(rec.get("mode", rec.get("timing/mode", "unknown")))].append(rec)

    payload: dict[str, Any] = {
        "schema_version": "cowp_v16_8_24_waymax_profile_summary_v1",
        "profiles": [str(p) for p in paths],
        "profile_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "scene_stats": {k: _stats(v) for k, v in scene_values.items()},
        "probe": {},
    }

    candidate_keys = (
        "rollout_seconds",
        "steps",
        "timing/env_reset_s",
        "timing/policy_build_s",
        "timing/action_s",
        "timing/env_step_s",
        "timing/metric_update_s",
        "timing/metric_OverlapMetric_s",
        "timing/metric_OffroadMetric_s",
        "timing/done_check_s",
        "timing/metric_finalize_s",
    )
    for mode, records in sorted(probe_by_mode.items()):
        by_key: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            for key in candidate_keys:
                v = _finite_float(rec.get(key))
                if v is not None:
                    by_key[key].append(v)
        mode_payload: dict[str, Any] = {
            "candidates": len(records),
            "stats": {k: _stats(v) for k, v in by_key.items()},
        }
        means = {k: statistics.fmean(v) for k, v in by_key.items() if v}
        denom = means.get("rollout_seconds", 0.0)
        if denom > 0:
            mode_payload["mean_fraction_of_candidate_wall"] = {
                k: float(v / denom)
                for k, v in means.items()
                if k.startswith("timing/") and k not in {"timing/mode"}
            }
        payload["probe"][mode] = mode_payload

    # Compact diagnostic hints; these do not change any replay settings.
    hints: list[str] = []
    dispatch = payload["probe"].get("dispatch", {}).get("stats", {})
    sync = payload["probe"].get("sync", {}).get("stats", {})
    def mean_of(stats: dict[str, Any], key: str) -> float:
        try:
            return float(stats.get(key, {}).get("mean", 0.0))
        except Exception:
            return 0.0
    if dispatch:
        d_done = mean_of(dispatch, "timing/done_check_s")
        d_wall = mean_of(dispatch, "rollout_seconds")
        if d_wall > 0 and d_done / d_wall > 0.25:
            hints.append("dispatch timing charges a large fraction to done_check; this is consistent with per-step host/device synchronization draining queued JAX work")
    if sync:
        components = {
            "env_step": mean_of(sync, "timing/env_step_s"),
            "OverlapMetric": mean_of(sync, "timing/metric_OverlapMetric_s"),
            "OffroadMetric": mean_of(sync, "timing/metric_OffroadMetric_s"),
            "action": mean_of(sync, "timing/action_s"),
            "done_check": mean_of(sync, "timing/done_check_s"),
        }
        if components:
            largest = max(components, key=components.get)
            hints.append(f"largest synchronization-aware timed stage: {largest} ({components[largest]:.6f} s/candidate mean)")
    payload["diagnostic_hints"] = hints

    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
