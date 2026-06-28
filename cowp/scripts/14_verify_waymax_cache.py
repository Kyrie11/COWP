from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _restore_key(k: str) -> str:
    return k.replace("__", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Waymax candidate-outcome fields in a COWP tensor cache.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if args.limit is not None:
        paths = paths[: int(args.limit)]
    scenes = 0
    scenes_with_any = 0
    rollout_valid_total = 0
    selected_total = 0
    collision_total = 0
    offroad_total = 0
    logdiv_values = []
    missing = 0
    for p in paths:
        scenes += 1
        with np.load(p, allow_pickle=True) as data:
            arrays = {_restore_key(k): data[k] for k in data.files}
        required = [
            "waymax/candidate_selected_for_rollout",
            "waymax/candidate_rollout_valid",
            "waymax/candidate_collision",
            "waymax/candidate_offroad",
            "waymax/candidate_log_divergence",
        ]
        if any(k not in arrays for k in required):
            missing += 1
            continue
        rv = np.asarray(arrays["waymax/candidate_rollout_valid"], dtype=bool)
        sel = np.asarray(arrays["waymax/candidate_selected_for_rollout"], dtype=bool)
        col = np.asarray(arrays["waymax/candidate_collision"], dtype=bool)
        off = np.asarray(arrays["waymax/candidate_offroad"], dtype=bool)
        ld = np.asarray(arrays["waymax/candidate_log_divergence"], dtype=np.float32)
        scenes_with_any += int(rv.any())
        rollout_valid_total += int(rv.sum())
        selected_total += int(sel.sum())
        collision_total += int((col & rv).sum())
        offroad_total += int((off & rv).sum())
        logdiv_values.extend(ld[rv & np.isfinite(ld)].astype(float).tolist())
    out = {
        "cache_dir": str(args.cache_dir),
        "scenes_checked": scenes,
        "scenes_missing_waymax_fields": missing,
        "scenes_with_any_rollout_valid": scenes_with_any,
        "rollout_valid_candidates": rollout_valid_total,
        "selected_for_rollout_candidates": selected_total,
        "collision_rate_among_valid_candidates": collision_total / max(rollout_valid_total, 1),
        "offroad_rate_among_valid_candidates": offroad_total / max(rollout_valid_total, 1),
        "mean_log_divergence_among_finite_valid_candidates": float(np.mean(logdiv_values)) if logdiv_values else None,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
