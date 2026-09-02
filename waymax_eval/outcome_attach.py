from __future__ import annotations

"""Incremental, atomic attachment of Waymax candidate outcomes to COWP cache NPZs.

This module is intentionally independent of the replay/JAX path so a GPU replay
worker can hand completed per-scene rows to a small background I/O worker.  The
resulting NPZ schema is the same as scripts/12_attach_waymax_candidate_outcomes.
"""

import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def restore_key(k: str) -> str:
    return k.replace("__", "/")


def store_key(k: str) -> str:
    return k.replace("/", "__")


def scenario_id(arrays: dict[str, np.ndarray], path: Path) -> str:
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


def bool_value(row: dict[str, Any], *names: str) -> bool:
    for name in names:
        if name in row:
            v = row[name]
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y"}
            return bool(v)
    return False


def row_rollout_valid(row: dict[str, Any]) -> bool:
    rv = bool_value(row, "rollout_valid", "candidate_rollout_valid", "valid")
    if not any(name in row for name in ("rollout_valid", "candidate_rollout_valid", "valid")):
        rv = not bool(row.get("error"))
    return bool(rv)


def float_value(row: dict[str, Any], *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                return float(row[name])
            except Exception:
                return default
    return default


def candidate_index(row: dict[str, Any]) -> int:
    for name in ("candidate_index", "candidate", "k", "candidate_id"):
        if name in row:
            return int(row[name])
    raise KeyError("outcome row missing candidate_index/candidate/k")


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray], *, compress: bool = False) -> None:
    """Write a complete NPZ to a temporary file and atomically rename it."""
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


def npz_readable(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            _ = data.files
        return True
    except Exception:
        return False


def npz_outcomes_complete_for_rows(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[bool, str]:
    rows = list(rows)
    if not path.exists():
        return False, "missing"
    if not rows:
        return npz_readable(path), "no_outcome_rows"
    try:
        with np.load(path, allow_pickle=True) as data:
            def get(key: str):
                stored = store_key(key)
                if stored in data.files:
                    return data[stored]
                if key in data.files:
                    return data[key]
                raise KeyError(key)

            selected = np.asarray(get("waymax/candidate_selected_for_rollout"), dtype=bool).reshape(-1)
            rollout_valid = np.asarray(get("waymax/candidate_rollout_valid"), dtype=bool).reshape(-1)
            collision = np.asarray(get("waymax/candidate_collision"), dtype=bool).reshape(-1)
            offroad = np.asarray(get("waymax/candidate_offroad"), dtype=bool).reshape(-1)
            logdiv = np.asarray(get("waymax/candidate_log_divergence"), dtype=np.float32).reshape(-1)
            k_count = int(selected.shape[0])
            if any(x.shape[0] != k_count for x in (rollout_valid, collision, offroad, logdiv)):
                return False, "shape_mismatch"
            for row in rows:
                k = candidate_index(row)
                if k < 0 or k >= k_count:
                    continue
                if not bool(selected[k]):
                    return False, f"missing_candidate:{k}"
                rv = row_rollout_valid(row)
                if bool(rollout_valid[k]) != rv:
                    return False, f"rollout_valid_mismatch:{k}"
                if rv:
                    col = bool_value(row, "collision", "candidate_collision", "overlap", "CollisionRate")
                    off = bool_value(row, "offroad", "candidate_offroad", "OffroadRate")
                    if bool(collision[k]) != col:
                        return False, f"collision_mismatch:{k}"
                    if bool(offroad[k]) != off:
                        return False, f"offroad_mismatch:{k}"
                    ld = float_value(row, "log_divergence", "candidate_log_divergence", "logdiv", "LogDivergence")
                    if np.isfinite(ld):
                        if not np.isfinite(logdiv[k]) or not np.isclose(float(logdiv[k]), float(ld), equal_nan=True):
                            return False, f"logdiv_mismatch:{k}"
    except Exception as exc:
        return False, f"load_error:{type(exc).__name__}"
    return True, "complete"


def attach_rows_to_cache_file(
    src: str | Path,
    dst: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    compress: bool = False,
    skip_if_complete: bool = True,
) -> dict[str, Any]:
    """Attach all currently-known rows for one scene and atomically write one NPZ.

    ``rows`` should contain the complete currently-known row set for the scene,
    not only rows produced by the latest resume attempt.  This guarantees that a
    resumed partial replay never drops outcomes that were already successful.
    """
    src = Path(src)
    dst = Path(dst)
    rows = list(rows)
    if skip_if_complete:
        ok, reason = npz_outcomes_complete_for_rows(dst, rows)
        if ok:
            return {"status": "skipped_complete", "src": str(src), "dst": str(dst), "rows": len(rows), "reason": reason}

    with np.load(src, allow_pickle=True) as data:
        arrays_raw = {k: data[k] for k in data.files}
    arrays = {restore_key(k): v for k, v in arrays_raw.items()}

    valid = np.asarray(arrays.get("cowp/candidates/valid", []), dtype=bool)
    k_count = int(valid.shape[0]) if valid.ndim >= 1 else 0
    selected = np.zeros(k_count, dtype=bool)
    rollout_valid = np.zeros(k_count, dtype=bool)
    collision = np.zeros(k_count, dtype=bool)
    offroad = np.zeros(k_count, dtype=bool)
    logdiv = np.full(k_count, np.nan, dtype=np.float32)
    seconds = np.full(k_count, np.nan, dtype=np.float32)
    row_count = 0
    valid_count = 0

    # Latest row wins per candidate, matching the replay/attach resume contract.
    by_k: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            by_k[candidate_index(row)] = row
        except Exception:
            continue
    for k, row in sorted(by_k.items()):
        if k < 0 or k >= k_count:
            continue
        row_count += 1
        selected[k] = True
        rv = row_rollout_valid(row)
        rollout_valid[k] = rv
        if rv:
            valid_count += 1
            collision[k] = bool_value(row, "collision", "candidate_collision", "overlap", "CollisionRate")
            offroad[k] = bool_value(row, "offroad", "candidate_offroad", "OffroadRate")
            logdiv[k] = float_value(row, "log_divergence", "candidate_log_divergence", "logdiv", "LogDivergence")
            seconds[k] = float_value(row, "rollout_seconds", "seconds", "candidate_rollout_seconds")

    arrays["waymax/candidate_selected_for_rollout"] = selected
    arrays["waymax/candidate_rollout_valid"] = rollout_valid
    arrays["waymax/candidate_collision"] = collision
    arrays["waymax/candidate_offroad"] = offroad
    arrays["waymax/candidate_log_divergence"] = logdiv
    arrays["waymax/candidate_rollout_seconds"] = seconds
    arrays["waymax/enabled"] = np.asarray(bool(rollout_valid.any()))
    arrays["waymax/outcomes_attached_count"] = np.asarray(int(row_count), dtype=np.int32)
    arrays["waymax/outcomes_valid_count"] = np.asarray(int(valid_count), dtype=np.int32)
    arrays["waymax/attach_complete_for_current_outcomes"] = np.asarray(bool(row_count == len(by_k)))
    arrays["waymax/rollout_status"] = np.asarray(
        "attached_real_waymax_outcomes" if rollout_valid.any() else (
            "attached_only_failed_waymax_outcomes" if row_count > 0 else "initialized_no_valid_outcomes"
        )
    )
    stored = {store_key(k): v for k, v in arrays.items()}
    write_npz_atomic(dst, stored, compress=compress)
    return {
        "status": "written",
        "src": str(src),
        "dst": str(dst),
        "scenario_id": scenario_id(arrays, src),
        "rows": int(row_count),
        "valid_rows": int(valid_count),
    }
