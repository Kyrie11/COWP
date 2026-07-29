#!/usr/bin/env python3
"""Inspect a staged v48 dedicated-calibration build without modifying it."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STAGES = {
    "safe": ("calibration_safe_w0", "calibration_safe_w1"),
    "near": ("calibration_near_w2", "calibration_near_w3"),
    "contact": ("calibration_contact_w4", "calibration_contact_w5"),
}
FINAL_ROOTS = ("calibration_safe", "calibration_near_contact", "calibration_contact")


def manifest_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return -1


def tail(path: Path, n: int = 12) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception as exc:
        return [f"<cannot read: {exc}>"]


def inspect(root: Path) -> dict:
    shards = root / "shards"
    logs = root / "logs"
    stage_info: dict[str, object] = {}
    first_incomplete: str | None = None
    for stage, names in STAGES.items():
        workers = []
        complete = True
        for name in names:
            count = manifest_count(shards / name / "manifest.csv")
            log_name = name.replace("calibration_near_", "calibration_near_") + ".log"
            log_path = logs / log_name
            workers.append({
                "name": name,
                "manifest_rows": count,
                "manifest_exists": count is not None,
                "log_exists": log_path.is_file(),
                "log_tail": tail(log_path),
            })
            complete &= count is not None and count >= 0
        stage_info[stage] = {"complete": bool(complete), "workers": workers}
        if first_incomplete is None and not complete:
            first_incomplete = stage

    final = {}
    finals_complete = True
    for name in FINAL_ROOTS:
        count = manifest_count(root / name / "manifest.csv")
        final[name] = {"manifest_rows": count, "complete": count is not None and count >= 0}
        finals_complete &= bool(final[name]["complete"])

    if first_incomplete is not None:
        recommended = first_incomplete
    elif not finals_complete:
        recommended = "merge"
    else:
        recommended = "complete"

    status_path = root / "calibration_build_status.json"
    status = None
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status = {"state": "unreadable", "detail": str(exc)}

    return {
        "root": str(root),
        "status_file": status,
        "stages": stage_info,
        "final_roots": final,
        "recommended_start_stage": recommended,
        "controller_log_tail": tail(logs / "calibration_controller.log", 20),
        "contact_logs_expected_now": bool(stage_info["near"]["complete"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()
    result = inspect(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))
    stage = result["recommended_start_stage"]
    if stage == "complete":
        print("\nRecommendation: build is complete; proceed to dedicated recalibration/evaluation.")
    else:
        print(f"\nRecommendation: resume with START_STAGE={stage} and RESUME=1 after confirming no controller is still running.")


if __name__ == "__main__":
    main()
