#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DIRECT_METRICS = (
    "closed_loop_FRA_exec",
    "closed_loop_DRS",
    "closed_loop_post_contact_deployability",
    "closed_loop_bounded_NUP",
    "closed_loop_audit_paper_pcd_selector_miss_rate",
    "closed_loop_audit_paper_selected_PCD_regret",
    "closed_loop_audit_selector_miss_rate",
    "intervention_rate",
    "intervention_episode_rate",
    "macro_switch_rate",
)
NESTED_METRICS = (
    "min_clearance_m_min",
    "min_clearance_m_p05",
    "ttc_s_min",
    "ttc_s_p05",
    "near_contact_exposure_rate",
    "critical_ttc_exposure_rate",
    "near_zero_clearance_exposure_rate",
    "overlap_episode_count",
    "secondary_overlap_event",
    "new_stable_stop_event",
    "time_to_stable_stop_steps",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _key(scene: dict[str, Any]) -> str:
    return str(scene.get("target_key") or f"{scene.get('bucket_name','')}|{scene.get('scene_id','')}|{scene.get('target_time_index','')}")


def _value(scene: dict[str, Any], name: str) -> float | None:
    if name in scene:
        value = scene.get(name)
    else:
        value = (scene.get("metric_summary") or {}).get(name)
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        means[i] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Scene-paired closed-loop comparison with bootstrap confidence intervals.")
    ap.add_argument("control", type=Path)
    ap.add_argument("method", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    control = _load(args.control)
    method = _load(args.method)
    c_scenes = {_key(s): s for s in control.get("scenes", [])}
    m_scenes = {_key(s): s for s in method.get("scenes", [])}
    common = sorted(set(c_scenes) & set(m_scenes))
    if not common:
        raise SystemExit("No paired scenes/targets found. Use results built from the same target list and seed.")

    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "control": str(args.control),
        "method": str(args.method),
        "num_control_scenes": len(c_scenes),
        "num_method_scenes": len(m_scenes),
        "num_paired_scenes": len(common),
        "metrics": {},
    }
    for name in DIRECT_METRICS + NESTED_METRICS:
        pairs = []
        for key in common:
            c = _value(c_scenes[key], name)
            m = _value(m_scenes[key], name)
            if c is not None and m is not None:
                pairs.append((c, m))
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        delta = arr[:, 1] - arr[:, 0]
        lo, hi = _bootstrap_ci(delta, rng, args.bootstrap)
        report["metrics"][name] = {
            "n": int(delta.size),
            "control_mean": float(np.mean(arr[:, 0])),
            "method_mean": float(np.mean(arr[:, 1])),
            "paired_delta": float(np.mean(delta)),
            "bootstrap_95ci": [lo, hi],
            "fraction_improved_raw": float(np.mean(delta > 0.0)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = args.output.with_suffix(".md")
    lines = [
        "# Paired closed-loop comparison",
        "",
        f"Paired scenes: **{len(common)}**",
        "",
        "| Metric | Control | Method | Paired delta | Bootstrap 95% CI | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["metrics"].items():
        lo, hi = row["bootstrap_95ci"]
        lines.append(f"| {name} | {row['control_mean']:.6f} | {row['method_mean']:.6f} | {row['paired_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {row['n']} |")
    lines += [
        "",
        "Positive delta is not universally better: for FRA, miss, exposure, intervention, overlap, and time-to-stop, lower is preferable.",
    ]
    md.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "markdown": str(md), "paired_scenes": len(common)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
