from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["scenario_id"]): r for r in d.get("scenario_results", [])}


def _collision(row: dict[str, Any]) -> bool:
    v = (row.get("standard_metrics", {}) or {}).get("CollisionRate")
    return bool(isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0)


def _metric(row: dict[str, Any], key: str) -> float | None:
    v = (row.get("standard_metrics", {}) or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Mandatory 19-scene early gate from the two frozen V29 counterexample conditions.")
    ap.add_argument("--cowp", required=True)
    ap.add_argument("--rvr", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cowp, rvr, method = _rows(_load(args.cowp)), _rows(_load(args.rvr)), _rows(_load(args.method))
    ids = [x.strip() for x in Path(args.ids).read_text().splitlines() if x.strip()]
    if len(ids) != 19 or len(set(ids)) != 19:
        raise SystemExit("gate19 manifest must contain exactly 19 unique IDs")
    if set(method) != set(ids):
        raise SystemExit(f"method result IDs do not match gate19 manifest: method={len(method)} manifest={len(ids)}")
    missing = [sid for sid in ids if sid not in cowp or sid not in rvr]
    if missing:
        raise SystemExit(f"reference rows missing gate19 IDs: {missing}")

    old_rescues = [sid for sid in ids if _collision(cowp[sid]) and not _collision(rvr[sid])]
    old_induced = [sid for sid in ids if not _collision(cowp[sid]) and _collision(rvr[sid])]
    if len(old_rescues) != 10 or len(old_induced) != 9 or set(old_rescues + old_induced) != set(ids):
        raise SystemExit(f"historical partition mismatch: rescues={len(old_rescues)} induced={len(old_induced)}")

    retained = [sid for sid in old_rescues if not _collision(method[sid])]
    lost = [sid for sid in old_rescues if _collision(method[sid])]
    avoided = [sid for sid in old_induced if not _collision(method[sid])]
    induced_remaining = [sid for sid in old_induced if _collision(method[sid])]
    mandatory_pass = len(retained) >= 5 and len(avoided) >= 7

    ep_delta = []
    kin_delta = 0
    for sid in ids:
        a, b = _metric(cowp[sid], "EP"), _metric(method[sid], "EP")
        if a is not None and b is not None:
            ep_delta.append(b - a)
        ka = bool((_metric(cowp[sid], "KinematicsInfeasibilityRate") or 0.0) > 0)
        kb = bool((_metric(method[sid], "KinematicsInfeasibilityRate") or 0.0) > 0)
        kin_delta += int(kb) - int(ka)

    out = {
        "schema_version": "v16.8.43r3_mandatory_gate19_v1",
        "classification": "early falsification only; cannot promote the algorithm and cannot replace full counterfactual48",
        "n": 19,
        "historical_rvr_rescue_ids": old_rescues,
        "historical_rvr_induced_ids": old_induced,
        "old_rvr_rescues_retained": len(retained),
        "old_rvr_rescues_lost_ids": lost,
        "old_rvr_induced_avoided": len(avoided),
        "old_rvr_induced_remaining_ids": induced_remaining,
        "frozen_thresholds": {"old_rvr_rescues_retained_min": 5, "old_rvr_induced_avoided_min": 7},
        "mandatory_collision_counterexample_gate": {
            "pass": bool(mandatory_pass),
            "failed": [
                *([] if len(retained) >= 5 else ["old_rvr_rescues_retained"]),
                *([] if len(avoided) >= 7 else ["old_rvr_induced_avoided"]),
            ],
        },
        "descriptive_only": {
            "mean_ep_delta_vs_cowp": (sum(ep_delta) / len(ep_delta)) if ep_delta else None,
            "kinematics_net_regression_scenes_vs_cowp": kin_delta,
        },
        "interpretation": (
            "If this gate fails, full counterfactual48 must fail the already-frozen six-item conjunction, so the remaining 29 scenes are unnecessary. "
            "If it passes, the remaining29 run is still mandatory because net collision, kinematics, EP and intervention gates require all 48 scenes."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
