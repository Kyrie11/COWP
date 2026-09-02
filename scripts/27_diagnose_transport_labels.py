from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset
from cowp.utils.progress import tqdm_iter


REQUIRED = (
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
    "cowp/transport/mode_retained_low_safe",
    "cowp/transport/response_root_index",
    "cowp/transport/response_is_min_burden",
    "cowp/transport/root_recovery_mass",
    "cowp/response/valid",
    "cowp/natural/weight",
    "cowp/witness/natural_conflict_mass",
    "cowp/witness/opr",
)


@dataclass
class FileStats:
    file: str
    ok: bool = False
    error: str | None = None
    mode_count: int = 0
    conflict_count: int = 0
    retain_count: int = 0
    root_valid: int = 0
    root_total: int = 0
    min_count: int = 0
    conflict_abs_sum: float = 0.0
    conflict_abs_count: int = 0
    opr_abs_sum: float = 0.0
    opr_abs_count: int = 0
    recovery_sum: float = 0.0
    recovery_count: int = 0
    recovery_out_of_range: int = 0
    recovery_hist: np.ndarray | None = None
    conflict_root_positive: int = 0
    conflict_root_total: int = 0
    conflict_root_confident: int = 0
    conflict_root_finite_min: int = 0
    legacy_witness_opr_abs_sum: float = 0.0
    legacy_witness_opr_abs_count: int = 0


def _quantile_from_hist(hist: np.ndarray, q: float, lo: float, hi: float) -> float | None:
    count = int(hist.sum())
    if count <= 0:
        return None
    target = max(1, int(np.ceil(float(q) * count)))
    idx = int(np.searchsorted(np.cumsum(hist, dtype=np.int64), target, side="left"))
    idx = min(max(idx, 0), len(hist) - 1)
    width = (hi - lo) / max(len(hist), 1)
    return float(lo + (idx + 0.5) * width)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check explicit same-root transport labels and their aggregate consistency.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="Threaded NPZ readers. Use 1 for strictly sequential I/O.")
    ap.add_argument("--quantile-bins", type=int, default=10000, help="Histogram bins for bounded-memory root-recovery quantiles.")
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    n = len(ds) if args.limit <= 0 else min(len(ds), int(args.limit))
    wanted = set(REQUIRED)
    wanted.update({
        "cowp/transport/root_low_safe_score",
        "cowp/transport/root_target_confidence",
        "cowp/transport/root_min_safe_burden",
        "cowp/transport/transported_opr",
        "cowp/transport/canonical_root_weight",
    })
    bins = max(int(args.quantile_bins), 100)
    q_lo, q_hi = 0.0, 1.0

    def inspect(idx: int) -> FileStats:
        path = ds.paths[idx]
        result = FileStats(file=path.name)
        try:
            data = ds.load(idx, wanted)
            missing = [k for k in REQUIRED if k not in data]
            if missing:
                result.error = f"missing keys: {missing}"
                return result
            mv = np.asarray(data["cowp/transport/mode_valid"], dtype=bool)
            mc = np.asarray(data["cowp/transport/mode_conflict"], dtype=bool)
            mr = np.asarray(data["cowp/transport/mode_retained_low_safe"], dtype=bool)
            rv = np.asarray(data["cowp/response/valid"], dtype=bool)
            ri = np.asarray(data["cowp/transport/response_root_index"], dtype=np.int64)
            rmin = np.asarray(data["cowp/transport/response_is_min_burden"], dtype=bool)
            rec = np.asarray(data["cowp/transport/root_recovery_mass"], dtype=np.float32)
            w = np.asarray(data["cowp/natural/weight"], dtype=np.float32)
            witness_conf = np.asarray(data["cowp/witness/natural_conflict_mass"], dtype=np.float32)
            witness_opr = np.asarray(data["cowp/witness/opr"], dtype=np.float32)

            result.mode_count = int(mv.sum())
            result.conflict_count = int((mc & mv).sum())
            result.retain_count = int((mr & mv).sum())
            result.root_total = int(rv.sum())
            result.root_valid = int((rv & (ri >= 0) & (ri < mv.shape[-1])).sum())
            result.min_count = int((rmin & rv).sum())

            finite_rec = rec[np.isfinite(rec)].astype(np.float64, copy=False).reshape(-1)
            if finite_rec.size:
                result.recovery_sum = float(finite_rec.sum(dtype=np.float64))
                result.recovery_count = int(finite_rec.size)
                result.recovery_out_of_range = int(((finite_rec < q_lo) | (finite_rec > q_hi)).sum())
                clipped = np.clip(finite_rec, q_lo, q_hi)
                result.recovery_hist = np.histogram(clipped, bins=bins, range=(q_lo, q_hi))[0].astype(np.int64)

            canonical_w = data.get("cowp/transport/canonical_root_weight")
            if canonical_w is not None:
                base_w = np.asarray(canonical_w, dtype=np.float32)
            else:
                # Existing v16.8 overlays did not persist the filtered/floor-smoothed
                # measure.  Preserve their stored self-consistency diagnostic here;
                # training/evaluation rebuild the canonical measure in
                # paper_aligned_supervision_batch.
                base_w = w
            ww = base_w[None, :, :]
            denom = (ww * mv).sum(axis=-1)
            conf = (ww * mv * mc).sum(axis=-1)
            root_score = np.asarray(
                data.get("cowp/transport/root_low_safe_score", np.zeros_like(mv, dtype=np.float32)),
                dtype=np.float32,
            )
            transported_root = np.where(mc, root_score, mr.astype(np.float32)) * mv.astype(np.float32)
            opr_reconstructed = np.clip((ww * transported_root).sum(axis=-1), 0.0, 1.0)
            opr_stored = np.asarray(
                data.get("cowp/transport/transported_opr", opr_reconstructed), dtype=np.float32
            )
            pair = denom > 1e-6
            if pair.any():
                conf_abs = np.abs(conf[pair] - witness_conf[pair]).astype(np.float64, copy=False)
                opr_abs = np.abs(opr_reconstructed[pair] - opr_stored[pair]).astype(np.float64, copy=False)
                legacy_abs = np.abs(opr_stored[pair] - witness_opr[pair]).astype(np.float64, copy=False)
                result.conflict_abs_sum = float(conf_abs.sum(dtype=np.float64))
                result.conflict_abs_count = int(conf_abs.size)
                result.opr_abs_sum = float(opr_abs.sum(dtype=np.float64))
                result.opr_abs_count = int(opr_abs.size)
                result.legacy_witness_opr_abs_sum = float(legacy_abs.sum(dtype=np.float64))
                result.legacy_witness_opr_abs_count = int(legacy_abs.size)

            conflict_roots = mv & mc
            result.conflict_root_total = int(conflict_roots.sum())
            result.conflict_root_positive = int(((root_score >= 0.35) & conflict_roots).sum())
            confidence = np.asarray(
                data.get("cowp/transport/root_target_confidence", np.zeros_like(root_score)), dtype=np.float32
            )
            result.conflict_root_confident = int(((confidence >= 0.25) & conflict_roots).sum())
            root_min = np.asarray(
                data.get("cowp/transport/root_min_safe_burden", np.full_like(root_score, 2.0)), dtype=np.float32
            )
            result.conflict_root_finite_min = int(((root_min < 2.0) & conflict_roots).sum())
            result.ok = True
            return result
        except Exception as exc:  # diagnostics must report the offending file
            result.error = repr(exc)
            return result

    indices = range(n)
    workers = max(1, int(args.workers))
    if workers == 1:
        results = map(inspect, indices)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(inspect, indices)

    files_ok = mode_count = conflict_count = retain_count = 0
    root_valid = root_total = min_count = 0
    conflict_sum = opr_sum = recovery_sum = legacy_opr_sum = 0.0
    conflict_n = opr_n = recovery_n = recovery_out_of_range = legacy_opr_n = 0
    conflict_root_positive = conflict_root_total = conflict_root_confident = conflict_root_finite_min = 0
    recovery_hist = np.zeros(bins, dtype=np.int64)
    errors: list[dict[str, str]] = []

    try:
        for row in tqdm_iter(results, total=n, desc="diagnose transport", unit="scene"):
            if not row.ok:
                errors.append({"file": row.file, "error": row.error or "unknown error"})
                continue
            files_ok += 1
            mode_count += row.mode_count
            conflict_count += row.conflict_count
            retain_count += row.retain_count
            root_valid += row.root_valid
            root_total += row.root_total
            min_count += row.min_count
            conflict_sum += row.conflict_abs_sum
            conflict_n += row.conflict_abs_count
            opr_sum += row.opr_abs_sum
            opr_n += row.opr_abs_count
            recovery_sum += row.recovery_sum
            recovery_n += row.recovery_count
            recovery_out_of_range += row.recovery_out_of_range
            conflict_root_positive += row.conflict_root_positive
            conflict_root_total += row.conflict_root_total
            conflict_root_confident += row.conflict_root_confident
            conflict_root_finite_min += row.conflict_root_finite_min
            legacy_opr_sum += row.legacy_witness_opr_abs_sum
            legacy_opr_n += row.legacy_witness_opr_abs_count
            if row.recovery_hist is not None:
                recovery_hist += row.recovery_hist
    finally:
        if workers != 1:
            pool.shutdown(wait=True)

    report = {
        "cache_dir": str(args.cache_dir),
        "files_total": n,
        "files_ok": files_ok,
        "error_count": len(errors),
        "workers": workers,
        "mode_valid_count": mode_count,
        "mode_conflict_rate": conflict_count / max(mode_count, 1),
        "mode_retained_low_safe_rate": retain_count / max(mode_count, 1),
        "response_root_assignment_coverage": root_valid / max(root_total, 1),
        "min_burden_marker_rate_per_valid_response": min_count / max(root_total, 1),
        "aggregate_conflict_mae": conflict_sum / conflict_n if conflict_n else None,
        "aggregate_opr_mae": opr_sum / opr_n if opr_n else None,
        "legacy_cached_witness_opr_mae": legacy_opr_sum / legacy_opr_n if legacy_opr_n else None,
        "conflict_root_positive_rate": conflict_root_positive / max(conflict_root_total, 1),
        "conflict_root_target_confidence_coverage": conflict_root_confident / max(conflict_root_total, 1),
        "conflict_root_safe_response_coverage": conflict_root_finite_min / max(conflict_root_total, 1),
        "conflict_root_count": conflict_root_total,
        "root_recovery_mean": recovery_sum / recovery_n if recovery_n else None,
        "root_recovery_p10": _quantile_from_hist(recovery_hist, 0.1, q_lo, q_hi),
        "root_recovery_p90": _quantile_from_hist(recovery_hist, 0.9, q_lo, q_hi),
        "root_recovery_count": recovery_n,
        "root_recovery_out_of_unit_interval": recovery_out_of_range,
        "root_recovery_quantile_method": "fixed_histogram",
        "root_recovery_quantile_bin_width": (q_hi - q_lo) / bins,
        "errors": errors[:20],
    }
    report["pass"] = bool(
        files_ok == n
        and mode_count > 0
        and report["response_root_assignment_coverage"] > 0.999
        and (report["aggregate_conflict_mae"] or 0.0) < 1e-3
        and (report["aggregate_opr_mae"] or 0.0) < 1e-3
        and recovery_out_of_range == 0
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
