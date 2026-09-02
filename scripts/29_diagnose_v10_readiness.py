from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _iter_sidecars(cache: Path):
    meta = json.loads((cache / "transport_augmentation_summary.json").read_text())
    side = cache / str(meta.get("sidecar_subdir", ".transport_v9"))
    yield from sorted(p for p in side.glob("*.npz") if not p.name.startswith("."))


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose v10 class priors and no-skill loss baselines")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    files = list(_iter_sidecars(Path(args.cache_dir)))
    if args.max_files > 0:
        files = files[: args.max_files]
    counts = {"valid": 0, "conflict": 0, "retain": 0, "pairs": 0, "recovery_positive": 0}
    recovery_sum = 0.0
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            mv = np.asarray(z["cowp/transport/mode_valid"], bool)
            mc = np.asarray(z["cowp/transport/mode_conflict"], bool)
            mr = np.asarray(z["cowp/transport/mode_retained_low_safe"], bool)
            rr = np.asarray(z["cowp/transport/root_recovery_mass"], float)
            counts["valid"] += int(mv.sum())
            counts["conflict"] += int((mc & mv).sum())
            counts["retain"] += int((mr & mv).sum())
            counts["pairs"] += int(rr.size)
            counts["recovery_positive"] += int((rr > 1e-4).sum())
            recovery_sum += float(np.nan_to_num(rr).sum())
    n=max(counts["valid"],1)
    pc=counts["conflict"]/n
    pr=counts["retain"]/n
    def entropy(p):
        p=min(max(p,1e-12),1-1e-12)
        return -(p*math.log(p)+(1-p)*math.log(1-p))
    out={
        "files_scanned": len(files),
        **counts,
        "mode_conflict_rate": pc,
        "mode_retain_rate": pr,
        "no_skill_bce_mode_conflict": entropy(pc),
        "no_skill_bce_mode_retain": entropy(pr),
        "root_recovery_mean": recovery_sum/max(counts["pairs"],1),
        "root_recovery_positive_rate": counts["recovery_positive"]/max(counts["pairs"],1),
        "interpretation": {
            "mode_loss_requirement": "validation BCE should be below the no-skill entropy baseline",
            "root_recovery": "very sparse labels require weighted presence supervision",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
