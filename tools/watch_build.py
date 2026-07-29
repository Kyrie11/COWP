#!/usr/bin/env python3
"""Watch an OC-RAP dataset build directory and report throughput/bottlenecks.

Usage:
  python tools/watch_build.py /path/to/output_dir --interval 10

It reads manifest.csv, dataset_summary.json and, when profiling is enabled,
build_profile.csv.  The script is read-only and safe to run in another terminal
while `python -m ocrap.cli build-dataset ...` is still running.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _summary(out: Path, window: int) -> str:
    manifest = _read_csv(out / "manifest.csv")
    profile = _read_csv(out / "build_profile.csv")
    ds = _read_json(out / "dataset_summary.json") or _read_json(out / "dataset_status.json")
    n_manifest = len(manifest)
    n_npz = len(list((out / "samples").glob("*.npz"))) if (out / "samples").exists() else 0
    lines = [f"samples: manifest={n_manifest} npz={n_npz} raw_scenarios={ds.get('raw_scenarios_seen', '?')} scene_times={ds.get('scene_time_groups', '?')}"]
    if profile:
        recent = profile[-max(1, int(window)):]
        cols = ["total_s", "future_generation_s", "teacher_margins_s", "root_clustering_s", "observation_s", "ocmero_s"]
        avg = {c: mean(_f(r, c) for r in recent) for c in cols}
        total = max(avg["total_s"], 1e-9)
        bottleneck = max(cols[1:], key=lambda c: avg[c])
        lines.append(
            "recent avg over last %d: total=%.2fs future=%.2fs teacher=%.2fs root=%.2fs obs=%.2fs ocmero=%.3fs"
            % (len(recent), avg["total_s"], avg["future_generation_s"], avg["teacher_margins_s"], avg["root_clustering_s"], avg["observation_s"], avg["ocmero_s"])
        )
        lines.append("bottleneck: %s (%.1f%% of sample time)" % (bottleneck, 100.0 * avg[bottleneck] / total))
        last = profile[-1]
        lines.append("last: scene=%s t=%s cand=%s macro=%s" % (last.get("scene_id"), last.get("time_index"), last.get("candidate_index"), last.get("macro_name")))
    else:
        lines.append("profiling CSV not found yet; run build with --set profiling.enabled=true to get stage timings")
    return " | ".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), _summary(args.output_dir, args.window), flush=True)
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
