from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cowp.utils.progress import tqdm_iter


def _get(z, key: str):
    return z[key.replace("/", "__")] if key.replace("/", "__") in z.files else z[key]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check explicit same-root transport labels and their aggregate consistency.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    paths = sorted(Path(args.cache_dir).glob("*.npz"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(args.cache_dir)

    files_ok = 0
    mode_count = conflict_count = retain_count = 0
    root_valid = root_total = min_count = 0
    conflict_abs = []
    opr_abs = []
    recovery_vals = []
    errors = []
    required = [
        "cowp/transport/mode_valid", "cowp/transport/mode_conflict",
        "cowp/transport/mode_retained_low_safe", "cowp/transport/response_root_index",
        "cowp/transport/response_is_min_burden", "cowp/transport/root_recovery_mass",
    ]
    for path in tqdm_iter(paths, desc="diagnose transport"):
        try:
            with np.load(path, allow_pickle=True) as z:
                if any(k.replace("/", "__") not in z.files and k not in z.files for k in required):
                    errors.append({"file": path.name, "error": "missing transport key"})
                    continue
                mv = np.asarray(_get(z, "cowp/transport/mode_valid"), dtype=bool)
                mc = np.asarray(_get(z, "cowp/transport/mode_conflict"), dtype=bool)
                mr = np.asarray(_get(z, "cowp/transport/mode_retained_low_safe"), dtype=bool)
                rv = np.asarray(_get(z, "cowp/response/valid"), dtype=bool)
                ri = np.asarray(_get(z, "cowp/transport/response_root_index"), dtype=np.int64)
                rmin = np.asarray(_get(z, "cowp/transport/response_is_min_burden"), dtype=bool)
                rec = np.asarray(_get(z, "cowp/transport/root_recovery_mass"), dtype=np.float32)
                w = np.asarray(_get(z, "cowp/natural/weight"), dtype=np.float32)
                witness_conf = np.asarray(_get(z, "cowp/witness/natural_conflict_mass"), dtype=np.float32)
                witness_opr = np.asarray(_get(z, "cowp/witness/opr"), dtype=np.float32)

                mode_count += int(mv.sum())
                conflict_count += int((mc & mv).sum())
                retain_count += int((mr & mv).sum())
                root_total += int(rv.sum())
                root_valid += int((rv & (ri >= 0) & (ri < mv.shape[-1])).sum())
                min_count += int((rmin & rv).sum())
                recovery_vals.extend(rec[np.isfinite(rec)].reshape(-1).tolist())

                ww = w[None, :, :]
                denom = (ww * mv).sum(axis=-1)
                conf = (ww * mv * mc).sum(axis=-1)
                opr = (ww * mv * mr).sum(axis=-1) / np.maximum(denom, 1e-6)
                opr = np.where(denom > 1e-6, opr, 1.0)
                pair = denom > 1e-6
                if pair.any():
                    conflict_abs.extend(np.abs(conf[pair] - witness_conf[pair]).tolist())
                    opr_abs.extend(np.abs(opr[pair] - witness_opr[pair]).tolist())
                files_ok += 1
        except Exception as exc:
            errors.append({"file": path.name, "error": repr(exc)})

    report = {
        "cache_dir": str(args.cache_dir),
        "files_total": len(paths), "files_ok": files_ok, "error_count": len(errors),
        "mode_valid_count": mode_count,
        "mode_conflict_rate": conflict_count / max(mode_count, 1),
        "mode_retained_low_safe_rate": retain_count / max(mode_count, 1),
        "response_root_assignment_coverage": root_valid / max(root_total, 1),
        "min_burden_marker_rate_per_valid_response": min_count / max(root_total, 1),
        "aggregate_conflict_mae": float(np.mean(conflict_abs)) if conflict_abs else None,
        "aggregate_opr_mae": float(np.mean(opr_abs)) if opr_abs else None,
        "root_recovery_mean": float(np.mean(recovery_vals)) if recovery_vals else None,
        "root_recovery_p10": float(np.quantile(recovery_vals, 0.1)) if recovery_vals else None,
        "root_recovery_p90": float(np.quantile(recovery_vals, 0.9)) if recovery_vals else None,
        "errors": errors[:20],
    }
    report["pass"] = bool(
        files_ok == len(paths) and mode_count > 0
        and report["response_root_assignment_coverage"] > 0.999
        and (report["aggregate_conflict_mae"] or 0.0) < 1e-3
        and (report["aggregate_opr_mae"] or 0.0) < 1e-3
    )
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
