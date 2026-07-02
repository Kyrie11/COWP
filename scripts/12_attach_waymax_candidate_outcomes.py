from __future__ import annotations

import argparse
import csv
import json
import glob
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _restore_key(k: str) -> str:
    return k.replace("__", "/")


def _store_key(k: str) -> str:
    return k.replace("/", "__")


def _scenario_id(arrays: dict[str, np.ndarray], path: Path) -> str:
    for key in ("scenario/id", "scenario__id", "womd/scenario/id", "womd__scenario__id"):
        if key in arrays:
            x = arrays[key]
            try:
                item = np.asarray(x).reshape(-1)[0]
                if isinstance(item, bytes):
                    return item.decode("utf-8")
                return str(item)
            except Exception:
                pass
    return path.stem


def _expand_outcome_paths(paths: str | list[str] | tuple[str, ...] | None) -> list[Path]:
    if not paths:
        return []
    if isinstance(paths, (list, tuple)):
        raw_items = [str(x) for x in paths]
    else:
        raw_items = [str(paths)]
    expanded: list[Path] = []
    for item in raw_items:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else []
            if matches:
                expanded.extend(Path(m) for m in matches)
            else:
                expanded.append(Path(part))
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    out: list[Path] = []
    for p in expanded:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _read_outcomes(paths: str | list[str] | tuple[str, ...] | None) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expanded = _expand_outcome_paths(paths)
    if not expanded:
        return rows
    missing = [str(p) for p in expanded if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Outcome file(s) not found: {missing[:5]}. This script only attaches existing replay results. "
            "First run scripts/13_replay_waymax_candidates with --outcomes-jsonl."
        )
    for p in expanded:
        if p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = str(row.get("scenario_id") or row.get("scenario/id") or "")
                    if sid:
                        rows[sid].append(row)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("scenario_id") or row.get("scenario/id") or "")
                if sid:
                    rows[sid].append(row)
    return rows


def _bool_value(row: dict[str, Any], *names: str) -> bool:
    for name in names:
        if name in row:
            v = row[name]
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y"}
            return bool(v)
    return False


def _float_value(row: dict[str, Any], *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                return float(row[name])
            except Exception:
                return default
    return default


def _candidate_index(row: dict[str, Any]) -> int:
    for name in ("candidate_index", "candidate", "k", "candidate_id"):
        if name in row:
            return int(row[name])
    raise KeyError("outcome row missing candidate_index/candidate/k")




def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray], *, compress: bool = False) -> None:
    """Write NPZ via temp file + atomic replace so readers never see partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _npz_readable(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            _ = data.files
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach Waymax candidate rollout outcomes to an existing COWP tensor cache without rebuilding WOMD labels.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--outcomes-jsonl", nargs="+", default=None, help="One or more JSONL/CSV outcome files, comma-separated paths, or shell globs with scenario_id,candidate_index,collision,offroad,log_divergence.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed. Default keeps --no-compress style for speed.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = _read_outcomes(args.outcomes_jsonl)
    paths = sorted(in_dir.glob("*.npz"))
    if args.limit is not None:
        paths = paths[: int(args.limit)]
    attached = 0
    copied = 0
    for src in paths:
        dst = out_dir / src.name
        if args.skip_existing and dst.exists() and _npz_readable(dst):
            continue
        with np.load(src, allow_pickle=True) as data:
            arrays_raw = {k: data[k] for k in data.files}
        arrays = {_restore_key(k): v for k, v in arrays_raw.items()}
        sid = _scenario_id(arrays, src)
        valid = np.asarray(arrays.get("cowp/candidates/valid", []), dtype=bool)
        K = int(valid.shape[0]) if valid.ndim >= 1 else 0
        selected = np.zeros(K, dtype=bool)
        rollout_valid = np.zeros(K, dtype=bool)
        collision = np.zeros(K, dtype=bool)
        offroad = np.zeros(K, dtype=bool)
        logdiv = np.full(K, np.nan, dtype=np.float32)
        seconds = np.full(K, np.nan, dtype=np.float32)
        for row in outcomes.get(sid, []):
            k = _candidate_index(row)
            if k < 0 or k >= K:
                continue
            selected[k] = True
            rv = _bool_value(row, "rollout_valid", "candidate_rollout_valid", "valid")
            # Backward-compatible: older outcome rows did not include rollout_valid;
            # if they contain no explicit error/status failure, treat the row as valid.
            if not any(name in row for name in ("rollout_valid", "candidate_rollout_valid", "valid")):
                rv = not bool(row.get("error"))
            rollout_valid[k] = bool(rv)
            if rollout_valid[k]:
                collision[k] = _bool_value(row, "collision", "candidate_collision", "overlap", "CollisionRate")
                offroad[k] = _bool_value(row, "offroad", "candidate_offroad", "OffroadRate")
                logdiv[k] = _float_value(row, "log_divergence", "candidate_log_divergence", "logdiv", "LogDivergence")
                seconds[k] = _float_value(row, "rollout_seconds", "seconds", "candidate_rollout_seconds")
        arrays["waymax/candidate_selected_for_rollout"] = selected
        arrays["waymax/candidate_rollout_valid"] = rollout_valid
        arrays["waymax/candidate_collision"] = collision
        arrays["waymax/candidate_offroad"] = offroad
        arrays["waymax/candidate_log_divergence"] = logdiv
        arrays["waymax/candidate_rollout_seconds"] = seconds
        arrays["waymax/enabled"] = np.asarray(bool(rollout_valid.any()))
        arrays["waymax/rollout_status"] = np.asarray("attached_real_waymax_outcomes" if rollout_valid.any() else "initialized_no_valid_outcomes")
        stored = {_store_key(k): v for k, v in arrays.items()}
        _write_npz_atomic(dst, stored, compress=bool(args.compress))
        attached += int(rollout_valid.any())
        copied += 1
    print(json.dumps({"processed": copied, "scenes_with_attached_outcomes": attached, "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
