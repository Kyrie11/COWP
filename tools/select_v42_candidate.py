#!/usr/bin/env python3
"""Select a v42 candidate only when calibration and offline gates are credible."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def method_result(path: Path) -> dict[str, Any] | None:
    d = load(path)
    if not d:
        return None
    return (d.get("methods", {}) or {}).get("ocrap")


def direct_reasons(r: dict[str, Any] | None) -> int:
    if not r:
        return 0
    return sum(int(v) for k, v in (r.get("selection_reason_counts", {}) or {}).items() if "direct_value" in str(k))


def f(r: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((r or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate(base: Path) -> tuple[list[str], float, dict[str, Any]]:
    offline = Path(str(base) + "_offline")
    failures: list[str] = []
    diag: dict[str, Any] = {"base_run": str(base), "offline_run": str(offline)}
    score = 0.0
    for bucket in ("near", "contact"):
        c = load(base / "calibration" / f"direct_value_advantage_{bucket}_v42.json")
        if not c:
            failures.append(f"{bucket}: missing calibration")
            continue
        groups = int(c.get("num_calibration_groups", 0))
        q = f(c, "direct_value_additive_q", float("inf"))
        challenge = f(c, "challenge_rate", 0.0)
        capture = f(c, "top1_opportunity_capture_rate", 0.0)
        precision = c.get("challenge_precision")
        neg = c.get("negative_challenge_rate")
        diag[f"{bucket}_cal"] = {
            "groups": groups,
            "q": q,
            "challenge_rate": challenge,
            "capture": capture,
            "precision": precision,
            "negative_challenge_rate": neg,
        }
        if groups < 30:
            failures.append(f"{bucket}: calibration groups < 30")
        if not math.isfinite(q):
            failures.append(f"{bucket}: non-finite q")
        if challenge < 0.005:
            failures.append(f"{bucket}: calibration challenge rate < 0.5%")
        if capture < 0.10:
            failures.append(f"{bucket}: top-1 opportunity capture < 10%")
        if precision is not None and float(precision) < 0.50:
            failures.append(f"{bucket}: challenge precision < 50%")
        if neg is not None and float(neg) > 0.10:
            failures.append(f"{bucket}: negative challenge rate > 10%")
        score += 2.0 * capture + challenge - 0.20 * q
        if precision is not None:
            score += 0.5 * float(precision)

    results: dict[str, dict[str, Any] | None] = {}
    for bucket in ("safe", "near_contact", "contact"):
        results[bucket] = method_result(offline / f"eval_{bucket}_v42_v42.json")
    if any(v is None for v in results.values()):
        failures.append("missing one or more offline v42 evaluations")
    safe, near, contact = results["safe"], results["near_contact"], results["contact"]
    if safe:
        if f(safe, "intervention_rate", 1.0) > 1e-12:
            failures.append("safe offline intervention is non-zero")
        if f(safe, "bounded_NUP", 0.0) < 0.999:
            failures.append("safe offline NUP < 0.999")
    if near and f(near, "bounded_NUP", 0.0) < 0.995:
        failures.append("near offline NUP < 0.995")
    if contact and f(contact, "bounded_NUP", 0.0) < 0.985:
        failures.append("contact offline NUP < 0.985")
    n_direct = direct_reasons(near) + direct_reasons(contact)
    diag["offline_direct_reasons"] = n_direct
    diag["offline"] = {
        b: None if r is None else {
            "FRA": r.get("FRA_exec"),
            "DRS": r.get("DRS"),
            "NUP": r.get("bounded_NUP"),
            "PCD": r.get("post_contact_deployability"),
            "intervention": r.get("intervention_rate"),
            "direct_reasons": direct_reasons(r),
        }
        for b, r in results.items()
    }
    if n_direct <= 0:
        failures.append("offline selector never used the v42 direct-value path")
    for r in (near, contact):
        if r:
            score += f(r, "post_contact_deployability", 0.0) - f(r, "FRA_exec", 1.0)
    return failures, score, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_runs", nargs="+", type=Path)
    ap.add_argument("--write-choice", type=Path, default=None)
    args = ap.parse_args()
    passed: list[tuple[float, Path]] = []
    for base in args.base_runs:
        failures, score, diag = evaluate(base)
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        if failures:
            for x in failures:
                print(f"FAIL {base.name}: {x}")
        else:
            print(f"PASS {base.name}: score={score:.6f}")
            passed.append((score, base))
    if not passed:
        print("RESULT: FAIL — no candidate should enter Waymax closed loop")
        return 2
    passed.sort(key=lambda x: (-x[0], str(x[1])))
    chosen = passed[0][1]
    print(f"RESULT: PASS — chosen {chosen}")
    if args.write_choice:
        args.write_choice.parent.mkdir(parents=True, exist_ok=True)
        args.write_choice.write_text(str(chosen) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
