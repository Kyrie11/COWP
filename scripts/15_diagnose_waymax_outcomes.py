from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _restore_key(k: str) -> str:
    return str(k).replace("__", "/")


def _expand_paths(items: list[str] | tuple[str, ...] | None) -> list[Path]:
    if not items:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else [part]
            if not matches:
                matches = [part]
            for m in matches:
                key = str(m)
                if key not in seen:
                    seen.add(key)
                    out.append(Path(m))
    return out


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "t"}:
            return True
        if s in {"0", "false", "no", "n", "f", "", "nan", "none"}:
            return False
        try:
            return float(s) != 0.0 and not math.isnan(float(s))
        except Exception:
            return bool(default)
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _as_float(v: Any, default: float = float("nan")) -> float:
    if v in (None, ""):
        return default
    try:
        return float(v)
    except Exception:
        return default


def _candidate_index(row: dict[str, Any]) -> int:
    for name in ("candidate_index", "candidate", "k", "candidate_id"):
        if name in row:
            try:
                return int(row[name])
            except Exception:
                return -1
    return -1


def _scenario_id_from_arrays(arrays: dict[str, np.ndarray], path: Path) -> str:
    for key in ("scenario/id", "womd/scenario/id"):
        if key in arrays:
            try:
                item = np.asarray(arrays[key]).reshape(-1)[0]
                if isinstance(item, bytes):
                    return item.decode("utf-8")
                if isinstance(item, np.bytes_):
                    return bytes(item).decode("utf-8")
                return str(item)
            except Exception:
                pass
    return path.stem


def _safe_rate(num: int | float, den: int | float) -> float | None:
    den = float(den)
    if den <= 0:
        return None
    return float(num) / den


def _finite_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def diagnose_outcomes(paths: list[Path]) -> dict[str, Any]:
    rows = 0
    malformed = 0
    duplicate_keys = 0
    keys: set[tuple[str, int]] = set()
    by_scene: dict[str, int] = defaultdict(int)
    valid = 0
    failed = 0
    collision = 0
    offroad = 0
    errors = Counter()
    steps: list[float] = []
    seconds: list[float] = []
    timing_values: dict[str, list[float]] = defaultdict(list)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                rows += 1
                sid = str(row.get("scenario_id") or row.get("scenario/id") or "")
                k = _candidate_index(row)
                if sid and k >= 0:
                    key = (sid, int(k))
                    duplicate_keys += int(key in keys)
                    keys.add(key)
                    by_scene[sid] += 1
                rv = _as_bool(row.get("rollout_valid", row.get("candidate_rollout_valid", row.get("valid"))), default=not bool(row.get("error")))
                valid += int(rv)
                failed += int(not rv)
                collision += int(rv and _as_bool(row.get("collision", row.get("candidate_collision", row.get("CollisionRate")))))
                offroad += int(rv and _as_bool(row.get("offroad", row.get("candidate_offroad", row.get("OffroadRate")))))
                if row.get("error"):
                    errors[str(row.get("error"))[:160]] += 1
                st = _as_float(row.get("steps"), default=float("nan"))
                if np.isfinite(st):
                    steps.append(st)
                sec = _as_float(row.get("rollout_seconds", row.get("candidate_rollout_seconds")), default=float("nan"))
                if np.isfinite(sec):
                    seconds.append(sec)
                for kk, vv in row.items():
                    if str(kk).startswith("timing/"):
                        x = _as_float(vv, default=float("nan"))
                        if np.isfinite(x):
                            timing_values[str(kk)].append(x)
    return {
        "outcome_files": [str(p) for p in paths],
        "rows": int(rows),
        "malformed_lines": int(malformed),
        "unique_scenario_candidate_keys": int(len(keys)),
        "duplicate_scenario_candidate_keys": int(duplicate_keys),
        "scenes_in_outcomes": int(len(by_scene)),
        "rows_per_scene": _finite_stats(by_scene.values()),
        "rollout_valid_rows": int(valid),
        "failed_rows": int(failed),
        "rollout_valid_rate": _safe_rate(valid, rows),
        "collision_rows_among_valid": int(collision),
        "offroad_rows_among_valid": int(offroad),
        "collision_rate_among_valid": _safe_rate(collision, valid),
        "offroad_rate_among_valid": _safe_rate(offroad, valid),
        "steps": _finite_stats(steps),
        "rollout_seconds": _finite_stats(seconds),
        "top_errors": errors.most_common(20),
        "candidate_timing_seconds": {k: _finite_stats(v) for k, v in sorted(timing_values.items())},
    }


def diagnose_attached_cache(cache_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    paths = sorted(cache_dir.glob("*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    required = (
        "waymax/candidate_selected_for_rollout",
        "waymax/candidate_rollout_valid",
        "waymax/candidate_collision",
        "waymax/candidate_offroad",
        "waymax/candidate_log_divergence",
    )
    scenes = 0
    missing = 0
    scenes_enabled = 0
    scenes_with_selected = 0
    scenes_with_valid = 0
    selected_total = 0
    rollout_valid_total = 0
    selected_invalid_total = 0
    rollout_not_selected_total = 0
    rollout_invalid_candidate_total = 0
    collision_total = 0
    offroad_total = 0
    collision_and_offroad_total = 0
    enabled_mismatch = 0
    status_counts = Counter()
    logdiv_values: list[float] = []
    seconds: list[float] = []

    # Cross-tabs against original COWP labels.
    buckets = {
        "valid": [0, 0, 0],
        "conventional_safe": [0, 0, 0],
        "false_safe": [0, 0, 0],
        "noncoercive_feasible": [0, 0, 0],
    }
    # entries are [rollout_valid_count, collision_count, offroad_count]

    for p in paths:
        scenes += 1
        with np.load(p, allow_pickle=True) as data:
            arrays = {_restore_key(k): data[k] for k in data.files}
        if any(k not in arrays for k in required):
            missing += 1
            continue
        valid_cand = np.asarray(arrays.get("cowp/candidates/valid", []), dtype=bool)
        K = int(valid_cand.shape[0]) if valid_cand.ndim >= 1 else 0
        sel = np.asarray(arrays["waymax/candidate_selected_for_rollout"], dtype=bool).reshape(-1)[:K]
        rv = np.asarray(arrays["waymax/candidate_rollout_valid"], dtype=bool).reshape(-1)[:K]
        col = np.asarray(arrays["waymax/candidate_collision"], dtype=bool).reshape(-1)[:K]
        off = np.asarray(arrays["waymax/candidate_offroad"], dtype=bool).reshape(-1)[:K]
        ld = np.asarray(arrays["waymax/candidate_log_divergence"], dtype=np.float32).reshape(-1)[:K]
        sec = np.asarray(arrays.get("waymax/candidate_rollout_seconds", np.full(K, np.nan)), dtype=np.float32).reshape(-1)[:K]
        enabled = _as_bool(np.asarray(arrays.get("waymax/enabled", False)).reshape(-1)[0] if "waymax/enabled" in arrays else False)
        status = arrays.get("waymax/rollout_status")
        if status is not None:
            try:
                item = np.asarray(status).reshape(-1)[0]
                status_counts[str(item.decode("utf-8") if isinstance(item, bytes) else item)] += 1
            except Exception:
                status_counts["<unreadable>"] += 1
        scenes_enabled += int(enabled)
        scenes_with_selected += int(sel.any())
        scenes_with_valid += int(rv.any())
        enabled_mismatch += int(enabled != bool(rv.any()))
        selected_total += int(sel.sum())
        rollout_valid_total += int(rv.sum())
        if valid_cand.size >= K and K > 0:
            selected_invalid_total += int((sel & ~valid_cand[:K]).sum())
            rollout_invalid_candidate_total += int((rv & ~valid_cand[:K]).sum())
        rollout_not_selected_total += int((rv & ~sel).sum())
        collision_total += int((col & rv).sum())
        offroad_total += int((off & rv).sum())
        collision_and_offroad_total += int((col & off & rv).sum())
        logdiv_values.extend(ld[rv & np.isfinite(ld)].astype(float).tolist())
        seconds.extend(sec[rv & np.isfinite(sec)].astype(float).tolist())

        label_masks = {
            "valid": valid_cand[:K] if valid_cand.size >= K else np.zeros(K, dtype=bool),
            "conventional_safe": np.asarray(arrays.get("cowp/candidates/conventional_safe", np.zeros(K, dtype=bool)), dtype=bool).reshape(-1)[:K],
            "false_safe": np.asarray(arrays.get("cowp/candidates/false_safe", np.zeros(K, dtype=bool)), dtype=bool).reshape(-1)[:K],
            "noncoercive_feasible": np.asarray(arrays.get("cowp/candidates/noncoercive_feasible", np.zeros(K, dtype=bool)), dtype=bool).reshape(-1)[:K],
        }
        for name, mask in label_masks.items():
            mask = mask & rv
            buckets[name][0] += int(mask.sum())
            buckets[name][1] += int((mask & col).sum())
            buckets[name][2] += int((mask & off).sum())

    bucket_summary = {}
    for name, (n, c, o) in buckets.items():
        bucket_summary[name] = {
            "rollout_valid": int(n),
            "collision_rate": _safe_rate(c, n),
            "offroad_rate": _safe_rate(o, n),
        }
    return {
        "cache_dir": str(cache_dir),
        "scenes_checked": int(scenes),
        "scenes_missing_waymax_fields": int(missing),
        "scenes_enabled": int(scenes_enabled),
        "scenes_with_selected_candidates": int(scenes_with_selected),
        "scenes_with_rollout_valid_candidates": int(scenes_with_valid),
        "selected_for_rollout_candidates": int(selected_total),
        "rollout_valid_candidates": int(rollout_valid_total),
        "selected_but_not_rollout_valid_candidates": int(max(selected_total - rollout_valid_total, 0)),
        "rollout_valid_but_not_selected_candidates": int(rollout_not_selected_total),
        "selected_invalid_original_candidates": int(selected_invalid_total),
        "rollout_valid_invalid_original_candidates": int(rollout_invalid_candidate_total),
        "enabled_flag_mismatched_scenes": int(enabled_mismatch),
        "collision_candidates_among_valid": int(collision_total),
        "offroad_candidates_among_valid": int(offroad_total),
        "collision_and_offroad_candidates_among_valid": int(collision_and_offroad_total),
        "collision_rate_among_valid_candidates": _safe_rate(collision_total, rollout_valid_total),
        "offroad_rate_among_valid_candidates": _safe_rate(offroad_total, rollout_valid_total),
        "log_divergence": _finite_stats(logdiv_values),
        "rollout_seconds": _finite_stats(seconds),
        "rollout_status_counts": status_counts.most_common(),
        "rates_by_original_cowp_label": bucket_summary,
    }


def diagnose_profile(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    sums = defaultdict(float)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and (k.endswith("_s") or k in {"seconds", "new_rows", "failed_rows", "resumed_rows", "candidates_selected"} or k.startswith("timing_sum/")):
                sums[k] += float(v)
    total_seconds = sums.get("seconds", 0.0)
    fractions = {k + "_fraction": (float(v) / total_seconds if total_seconds > 0 and k.endswith("_s") else None) for k, v in sums.items()}
    out = {
        "profile_files": [str(p) for p in paths],
        "scene_rows": len(rows),
        "sums": dict(sorted(sums.items())),
        "fractions_of_profile_seconds": {k: v for k, v in sorted(fractions.items()) if v is not None},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose Waymax candidate replay JSONL/profile and attached waymax/* cache fields.")
    ap.add_argument("--cache-dir", default=None, help="Attached tensor cache directory containing waymax/* fields.")
    ap.add_argument("--outcomes-jsonl", nargs="*", default=None, help="Outcome JSONL path(s), comma-separated paths, or shell globs.")
    ap.add_argument("--profile-jsonl", nargs="*", default=None, help="Replay profile JSONL path(s), comma-separated paths, or shell globs.")
    ap.add_argument("--limit", type=int, default=None, help="Limit attached cache files when --cache-dir is used.")
    ap.add_argument("--output-json", default=None, help="Optional path to save the diagnostic JSON.")
    args = ap.parse_args()

    result: dict[str, Any] = {}
    if args.outcomes_jsonl:
        result["outcomes"] = diagnose_outcomes(_expand_paths(args.outcomes_jsonl))
    if args.cache_dir:
        result["attached_cache"] = diagnose_attached_cache(Path(args.cache_dir), limit=args.limit)
    if args.profile_jsonl:
        result["profile"] = diagnose_profile(_expand_paths(args.profile_jsonl))
    if not result:
        raise SystemExit("Provide at least one of --cache-dir, --outcomes-jsonl, or --profile-jsonl")
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
