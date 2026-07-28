from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify per-mode transport labels and their aggregate witness identities.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-files", type=int, default=512)
    ap.add_argument("--atol", type=float, default=2e-4)
    args = ap.parse_args()

    paths = sorted(Path(args.cache_dir).glob("*.npz"))[: max(int(args.max_files), 0)]
    if not paths:
        raise FileNotFoundError(f"No NPZ files in {args.cache_dir}")
    checked = 0
    missing = 0
    logical_errors = 0
    max_conflict_err = 0.0
    max_opr_err = 0.0
    support_pairs = 0
    mode_support_rate = []
    required = (
        "cowp/transport/mode_support",
        "cowp/transport/mode_conflict",
        "cowp/transport/mode_retained",
    )
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            if not all(k in z.files for k in required):
                missing += 1
                continue
            support = np.asarray(z[required[0]], dtype=bool)
            conflict = np.asarray(z[required[1]], dtype=bool)
            retained = np.asarray(z[required[2]], dtype=bool)
            weight = np.asarray(z["cowp/natural/weight"], dtype=np.float64)
            cand_valid = np.asarray(z["cowp/candidates/valid"], dtype=bool)
            crit_valid = np.asarray(z["cowp/critical/valid"], dtype=bool)
            if np.any(conflict & retained):
                logical_errors += 1
            if np.any(conflict & ~support[None, ...]) or np.any(retained & ~support[None, ...]):
                logical_errors += 1
            pair = cand_valid[:, None] & crit_valid[None, :]
            pred_conflict = (weight[None, :, :] * conflict).sum(axis=-1)
            target_conflict = np.asarray(z["cowp/witness/natural_conflict_mass"], dtype=np.float64)
            if np.any(pair):
                max_conflict_err = max(max_conflict_err, float(np.max(np.abs(pred_conflict[pair] - target_conflict[pair]))))
            low_mass = (weight * support).sum(axis=-1)
            retained_mass = (weight[None, :, :] * retained).sum(axis=-1)
            pred_opr = np.where(low_mass[None, :] > 1e-8, retained_mass / np.maximum(low_mass[None, :], 1e-8), 1.0)
            target_opr = np.asarray(z["cowp/witness/opr"], dtype=np.float64)
            if np.any(pair):
                max_opr_err = max(max_opr_err, float(np.max(np.abs(pred_opr[pair] - target_opr[pair]))))
            support_pairs += int((pair & (low_mass[None, :] > 1e-8)).sum())
            mode_support_rate.append(float(support.mean()))
            checked += 1
    report = {
        "files_considered": len(paths),
        "files_checked": checked,
        "files_missing_transport": missing,
        "logical_error_files": logical_errors,
        "max_conflict_mass_reconstruction_error": max_conflict_err,
        "max_opr_reconstruction_error": max_opr_err,
        "pairs_with_supported_natural_set": support_pairs,
        "mean_mode_support_rate": float(np.mean(mode_support_rate)) if mode_support_rate else 0.0,
        "atol": float(args.atol),
    }
    report["pass"] = bool(
        checked > 0 and missing == 0 and logical_errors == 0
        and max_conflict_err <= args.atol and max_opr_err <= args.atol
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
