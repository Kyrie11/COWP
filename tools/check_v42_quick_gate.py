#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def metric(d: dict[str, Any], key: str, default: float | None = None) -> float | None:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def direct_count(d: dict[str, Any]) -> int:
    return sum(int(v) for k, v in (d.get("selection_reason_counts", {}) or {}).items() if "direct_value" in str(k))


def _scene_records(result_path: Path, summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    embedded = summary.get("scenes", []) or []
    for scene in embedded:
        if isinstance(scene, dict):
            yield scene
    jsonl = Path(str(result_path) + ".scenes.jsonl")
    if not jsonl.exists():
        return
    with jsonl.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene = row.get("scene", row)
            if isinstance(scene, dict):
                yield scene


def intervention_diagnostics(result_path: Path, summary: dict[str, Any], threshold: float) -> tuple[int, list[tuple[Any, ...]], int]:
    total = 0
    bad: list[tuple[Any, ...]] = []
    direct = 0
    for scene in _scene_records(result_path, summary):
        for decision in scene.get("decisions", []) or []:
            macro = str(decision.get("selected_macro", "nominal") or "nominal").lower()
            if macro in {"nominal", "keep", ""}:
                continue
            total += 1
            reason = str(decision.get("selection_reason", ""))
            direct += int("direct_value" in reason)
            dev = float(decision.get("selected_nominal_deviation") or 0.0)
            if dev < threshold:
                bad.append((scene.get("scene_id"), decision.get("step_index"), dev, reason, macro))
    return total, bad, direct


def _calibration_dir(base_run: Path | None, run: Path) -> Path:
    if base_run is not None:
        return base_run if base_run.name == "calibration" else base_run / "calibration"
    stem = run.name
    for suffix in ("_eval", "_offline", "_micro", "_confirm12"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return run.parent / stem / "calibration"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--base-run", type=Path, default=None)
    ap.add_argument("--min-deviation", type=float, default=0.002)
    ap.add_argument("--min-calibration-challenge-rate", type=float, default=0.005)
    ap.add_argument("--min-opportunity-capture", type=float, default=0.10)
    args = ap.parse_args()

    root = args.run
    cal_dir = _calibration_dir(args.base_run, root)
    failures: list[str] = []
    warnings: list[str] = []
    rows: list[tuple[str, Any]] = []

    for bucket in ("near", "contact"):
        paths = list(root.glob(f"**/direct_value_advantage_{bucket}_v42.json"))
        paths += [cal_dir / f"direct_value_advantage_{bucket}_v42.json"]
        c = next((load(p) for p in paths if p.exists() and load(p) is not None), None)
        if c is None:
            failures.append(f"{bucket}: missing v42 selection-conditional calibration")
            continue
        groups = int(c.get("num_calibration_groups", 0))
        q = float(c.get("direct_value_additive_q", float("inf")))
        challenge = float(c.get("challenge_rate") or 0.0)
        capture = float(c.get("top1_opportunity_capture_rate") or 0.0)
        neg = c.get("negative_challenge_rate")
        rows.extend(
            [
                (f"cal_{bucket}_groups", groups),
                (f"cal_{bucket}_q", q),
                (f"cal_{bucket}_challenge_rate", challenge),
                (f"cal_{bucket}_opportunity_capture", capture),
                (f"cal_{bucket}_challenge_precision", c.get("challenge_precision")),
            ]
        )
        if groups < 30:
            failures.append(f"{bucket}: fewer than 30 calibration groups")
        if not math.isfinite(q):
            failures.append(f"{bucket}: non-finite additive q")
        if challenge < args.min_calibration_challenge_rate:
            failures.append(f"{bucket}: calibrated top-1 challenge rate < {args.min_calibration_challenge_rate:.3f}")
        if capture < args.min_opportunity_capture:
            failures.append(f"{bucket}: top-1 opportunity capture < {args.min_opportunity_capture:.2f}")
        if neg is not None and float(neg) > 0.10:
            failures.append(f"{bucket}: negative challenge rate > 10%")

    safe_path = root / "eval_safe_v42_v42.json"
    safe = load(safe_path)
    if safe:
        result = (safe.get("methods", {}) or {}).get("ocrap", {})
        if float(result.get("intervention_rate", 1.0)) > 1e-12:
            failures.append("safe offline intervention is non-zero")
        if float(result.get("bounded_NUP", 0.0)) < 0.999:
            failures.append("safe offline NUP < 0.999")

    for bucket in ("near_contact", "contact"):
        v_path = root / f"audit_{bucket}_selected_topk_v42_v42.json"
        s_path = root / f"audit_{bucket}_selected_topk_v42_scalar.json"
        v = load(v_path)
        s = load(s_path)
        if not v:
            continue
        total, bad, direct_decisions = intervention_diagnostics(v_path, v, args.min_deviation)
        rows.extend(
            [
                (f"{bucket}_decisions", v.get("num_decisions")),
                (f"{bucket}_pcd", v.get("closed_loop_post_contact_deployability")),
                (f"{bucket}_paper_regret", v.get("closed_loop_audit_paper_selected_PCD_regret")),
                (f"{bucket}_nup", v.get("closed_loop_bounded_NUP")),
                (f"{bucket}_interventions", total),
                (f"{bucket}_direct_interventions", direct_decisions),
            ]
        )
        if bad:
            failures.append(f"{bucket}: {len(bad)} intervention(s) below actionability threshold")
        nup_floor = 0.995 if bucket == "near_contact" else 0.985
        int_cap = 0.04 if bucket == "near_contact" else 0.08
        if (metric(v, "closed_loop_bounded_NUP", 0.0) or 0.0) < nup_floor:
            failures.append(f"{bucket}: NUP below {nup_floor}")
        if (metric(v, "intervention_rate", 1.0) or 1.0) > int_cap:
            failures.append(f"{bucket}: intervention rate > {int_cap}")
        if direct_decisions == 0:
            failures.append(f"{bucket}: OCSAVA direct path never entered final decisions")
        if total > 0 and direct_decisions < total:
            warnings.append(f"{bucket}: {total - direct_decisions} intervention(s) came from non-v42 paths")
        if s:
            vp = metric(v, "closed_loop_post_contact_deployability", 0.0) or 0.0
            sp = metric(s, "closed_loop_post_contact_deployability", 0.0) or 0.0
            vr = metric(v, "closed_loop_audit_paper_selected_PCD_regret", 1.0) or 1.0
            sr = metric(s, "closed_loop_audit_paper_selected_PCD_regret", 1.0) or 1.0
            if vp + 0.005 < sp:
                failures.append(f"{bucket}: PCD worse than paired scalar by >0.005")
            if vr > sr + 0.005:
                failures.append(f"{bucket}: paper PCD regret worse than scalar by >0.005")
            # A development gate needs directional physical evidence, not merely
            # non-degradation.  Either PCD or regret must improve measurably.
            if (vp - sp) < 0.005 and (sr - vr) < 0.005:
                failures.append(f"{bucket}: no >=0.005 directional PCD/regret improvement")

    print("V42 OCSAVA QUICK GATE")
    print(f"calibration_dir: {cal_dir}")
    for key, value in rows:
        print(f"{key}: {value}")
    for warning in warnings:
        print("WARNING:", warning)
    for failure in failures:
        print("FAIL:", failure)
    if failures:
        print("RESULT: FAIL — do not expand")
        return 2
    print("RESULT: PASS — proceed to 12-rollout confirmation, not publication scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
