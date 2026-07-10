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


def _row_key(row: dict[str, Any]) -> tuple[str, int] | None:
    sid = str(row.get("scenario_id") or row.get("scenario/id") or "")
    if not sid:
        return None
    try:
        k = _candidate_index(row)
    except Exception:
        return None
    if k < 0:
        return None
    return sid, int(k)


def _read_outcomes(
    paths: str | list[str] | tuple[str, ...] | None,
    *,
    repair_jsonl: bool = False,
    allow_corrupt_tail: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Read outcome rows grouped by scenario id.

    Rows are deduplicated by (scenario_id, candidate_index), keeping the latest
    complete row in the order the input files are supplied.  For JSONL files,
    --repair-outcomes-jsonl rewrites each file to complete JSON objects only;
    --allow-corrupt-tail only ignores malformed lines while reading.
    """
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    key_order: list[tuple[str, int]] = []
    stats = {
        "outcome_files": 0,
        "outcome_lines": 0,
        "outcome_rows": 0,
        "outcome_dropped_empty": 0,
        "outcome_dropped_corrupt": 0,
        "outcome_dropped_keyless": 0,
        "outcome_deduplicated": 0,
        "outcome_repaired_files": 0,
    }
    expanded = _expand_outcome_paths(paths)
    if not expanded:
        return defaultdict(list), stats
    missing = [str(p) for p in expanded if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Outcome file(s) not found: {missing[:5]}. This script only attaches existing replay results. "
            "First run scripts/13_replay_waymax_candidates with --outcomes-jsonl."
        )

    for p in expanded:
        stats["outcome_files"] += 1
        file_rows: list[dict[str, Any]] = []
        file_keys: list[tuple[str, int]] = []
        if p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stats["outcome_lines"] += 1
                    key = _row_key(row)
                    if key is None:
                        stats["outcome_dropped_keyless"] += 1
                        continue
                    file_rows.append(dict(row))
                    file_keys.append(key)
            # CSV repair is intentionally not attempted.
        else:
            raw_lines: list[str] = []
            with p.open("r", encoding="utf-8") as f:
                for raw in f:
                    stats["outcome_lines"] += 1
                    line = raw.strip()
                    if not line:
                        stats["outcome_dropped_empty"] += 1
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        stats["outcome_dropped_corrupt"] += 1
                        if allow_corrupt_tail or repair_jsonl:
                            continue
                        raise
                    if not isinstance(row, dict):
                        stats["outcome_dropped_keyless"] += 1
                        continue
                    key = _row_key(row)
                    if key is None:
                        stats["outcome_dropped_keyless"] += 1
                        continue
                    file_rows.append(row)
                    file_keys.append(key)
                    raw_lines.append(json.dumps(row, ensure_ascii=False, allow_nan=True))

            if repair_jsonl:
                needs_rewrite = bool(stats["outcome_dropped_empty"] or stats["outcome_dropped_corrupt"] or stats["outcome_dropped_keyless"])
                try:
                    if p.stat().st_size > 0:
                        with p.open("rb") as bf:
                            bf.seek(-1, 2)
                            needs_rewrite = needs_rewrite or (bf.read(1) != b"\n")
                except OSError:
                    needs_rewrite = True
                if needs_rewrite:
                    tmp = p.with_name(f"{p.name}.attach_repair.{os.getpid()}.tmp")
                    with tmp.open("w", encoding="utf-8") as f:
                        for line in raw_lines:
                            f.write(line + "\n")
                    tmp.replace(p)
                    stats["outcome_repaired_files"] += 1

        for key, row in zip(file_keys, file_rows):
            if key in rows_by_key:
                stats["outcome_deduplicated"] += 1
                try:
                    key_order.remove(key)
                except ValueError:
                    pass
            key_order.append(key)
            rows_by_key[key] = row

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in key_order:
        row = rows_by_key[key]
        rows[key[0]].append(row)
    stats["outcome_rows"] = len(rows_by_key)
    return rows, stats


def _bool_value(row: dict[str, Any], *names: str) -> bool:
    for name in names:
        if name in row:
            v = row[name]
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y"}
            return bool(v)
    return False


def _row_rollout_valid(row: dict[str, Any]) -> bool:
    rv = _bool_value(row, "rollout_valid", "candidate_rollout_valid", "valid")
    # Backward-compatible: older outcome rows did not include rollout_valid;
    # if they contain no explicit error/status failure, treat the row as valid.
    if not any(name in row for name in ("rollout_valid", "candidate_rollout_valid", "valid")):
        rv = not bool(row.get("error"))
    return bool(rv)


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


def _npz_outcomes_complete_for_rows(path: Path, rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return whether an existing attached NPZ already covers these outcome rows.

    This is the incremental-attach guard for --skip-existing: a file is skipped
    only when all candidate rows currently available in the outcome JSONL are
    already reflected in waymax/candidate_selected_for_rollout and related
    outcome arrays.  Placeholder files written from an earlier partial JSONL
    therefore get overwritten when new rows become available.
    """
    if not path.exists():
        return False, "missing"
    if not rows:
        return _npz_readable(path), "no_outcome_rows"
    try:
        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            key_selected = _store_key("waymax/candidate_selected_for_rollout")
            key_valid = _store_key("waymax/candidate_rollout_valid")
            key_collision = _store_key("waymax/candidate_collision")
            key_offroad = _store_key("waymax/candidate_offroad")
            key_logdiv = _store_key("waymax/candidate_log_divergence")
            for k in (key_selected, key_valid, key_collision, key_offroad, key_logdiv):
                if k not in files and _restore_key(k) not in files:
                    return False, f"missing_key:{_restore_key(k)}"

            def get(key: str):
                stored = _store_key(key)
                return data[stored] if stored in data.files else data[key]

            selected = np.asarray(get("waymax/candidate_selected_for_rollout"), dtype=bool).reshape(-1)
            rollout_valid = np.asarray(get("waymax/candidate_rollout_valid"), dtype=bool).reshape(-1)
            collision = np.asarray(get("waymax/candidate_collision"), dtype=bool).reshape(-1)
            offroad = np.asarray(get("waymax/candidate_offroad"), dtype=bool).reshape(-1)
            logdiv = np.asarray(get("waymax/candidate_log_divergence"), dtype=np.float32).reshape(-1)
            K = int(selected.shape[0])
            if rollout_valid.shape[0] != K:
                return False, "shape_mismatch:rollout_valid"
            for row in rows:
                k = _candidate_index(row)
                if k < 0:
                    continue
                if k >= K:
                    # The source candidate count may have changed; reattaching
                    # from the source cache will ignore the out-of-range row in
                    # the same way as the normal attach path.
                    continue
                if not bool(selected[k]):
                    return False, f"missing_candidate:{k}"
                rv = _row_rollout_valid(row)
                if bool(rollout_valid[k]) != bool(rv):
                    return False, f"rollout_valid_mismatch:{k}"
                if rv:
                    col = _bool_value(row, "collision", "candidate_collision", "overlap", "CollisionRate")
                    off = _bool_value(row, "offroad", "candidate_offroad", "OffroadRate")
                    if bool(collision[k]) != bool(col):
                        return False, f"collision_mismatch:{k}"
                    if bool(offroad[k]) != bool(off):
                        return False, f"offroad_mismatch:{k}"
                    row_ld = _float_value(row, "log_divergence", "candidate_log_divergence", "logdiv", "LogDivergence")
                    if np.isfinite(row_ld):
                        if not np.isfinite(logdiv[k]) or not np.isclose(float(logdiv[k]), float(row_ld), equal_nan=True):
                            return False, f"logdiv_mismatch:{k}"
    except Exception as exc:
        return False, f"load_error:{type(exc).__name__}"
    return True, "complete"


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach Waymax candidate rollout outcomes to an existing COWP tensor cache without rebuilding WOMD labels.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--outcomes-jsonl", nargs="+", default=None, help="One or more JSONL/CSV outcome files, comma-separated paths, or shell globs with scenario_id,candidate_index,collision,offroad,log_divergence.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed. Default keeps --no-compress style for speed.")
    ap.add_argument("--skip-existing", action="store_true", help="Incremental-safe skip: skip only files whose attached Waymax fields already cover the currently available outcome rows.")
    ap.add_argument("--legacy-file-skip-existing", action="store_true", help="Old behavior: with --skip-existing, skip any readable output NPZ by filename only. Not safe for partial-then-final attach.")
    ap.add_argument("--only-with-outcomes", action="store_true", help="Write only scenes that currently have at least one outcome row. Useful for a small partial training cache; final attach should omit this flag.")
    ap.add_argument("--allow-corrupt-tail", action="store_true", help="Ignore malformed JSONL lines while reading outcomes. Prefer pausing replay or --repair-outcomes-jsonl when possible.")
    ap.add_argument("--repair-outcomes-jsonl", action="store_true", help="Rewrite JSONL outcome files to complete, newline-terminated, deduplicated rows before attach.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes, outcome_stats = _read_outcomes(
        args.outcomes_jsonl,
        repair_jsonl=bool(args.repair_outcomes_jsonl),
        allow_corrupt_tail=bool(args.allow_corrupt_tail),
    )
    paths = sorted(in_dir.glob("*.npz"))
    if args.limit is not None:
        paths = paths[: int(args.limit)]

    attached = 0
    copied = 0
    skipped_existing = 0
    skipped_no_outcomes = 0
    rewritten_existing = 0
    initialized_no_outcomes = 0
    for src in paths:
        # Fast sid path: the replay code also uses filename stem by default.
        sid_hint = src.stem
        sid_rows = outcomes.get(sid_hint, [])
        dst = out_dir / src.name
        if args.only_with_outcomes and not sid_rows:
            skipped_no_outcomes += 1
            continue
        if args.skip_existing and dst.exists():
            if args.legacy_file_skip_existing:
                if _npz_readable(dst):
                    skipped_existing += 1
                    continue
            else:
                ok, _reason = _npz_outcomes_complete_for_rows(dst, sid_rows)
                if ok:
                    skipped_existing += 1
                    continue
                if _npz_readable(dst):
                    rewritten_existing += 1

        with np.load(src, allow_pickle=True) as data:
            arrays_raw = {k: data[k] for k in data.files}
        arrays = {_restore_key(k): v for k, v in arrays_raw.items()}
        sid = _scenario_id(arrays, src)
        # If the true id differs from the filename, prefer the true id.  This
        # keeps --verify-cache-sid style caches correct while preserving the
        # fast path above for normal COWP files.
        if sid != sid_hint:
            sid_rows = outcomes.get(sid, sid_rows)
            if args.only_with_outcomes and not sid_rows:
                skipped_no_outcomes += 1
                continue
            if args.skip_existing and dst.exists() and not args.legacy_file_skip_existing:
                ok, _reason = _npz_outcomes_complete_for_rows(dst, sid_rows)
                if ok:
                    skipped_existing += 1
                    continue

        valid = np.asarray(arrays.get("cowp/candidates/valid", []), dtype=bool)
        K = int(valid.shape[0]) if valid.ndim >= 1 else 0
        selected = np.zeros(K, dtype=bool)
        rollout_valid = np.zeros(K, dtype=bool)
        collision = np.zeros(K, dtype=bool)
        offroad = np.zeros(K, dtype=bool)
        logdiv = np.full(K, np.nan, dtype=np.float32)
        seconds = np.full(K, np.nan, dtype=np.float32)
        row_count = 0
        valid_count = 0
        for row in sid_rows:
            k = _candidate_index(row)
            if k < 0 or k >= K:
                continue
            row_count += 1
            selected[k] = True
            rv = _row_rollout_valid(row)
            rollout_valid[k] = bool(rv)
            if rollout_valid[k]:
                valid_count += 1
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
        arrays["waymax/outcomes_attached_count"] = np.asarray(int(row_count), dtype=np.int32)
        arrays["waymax/outcomes_valid_count"] = np.asarray(int(valid_count), dtype=np.int32)
        arrays["waymax/attach_complete_for_current_outcomes"] = np.asarray(bool(row_count == len(sid_rows)))
        arrays["waymax/rollout_status"] = np.asarray(
            "attached_real_waymax_outcomes" if rollout_valid.any() else (
                "attached_only_failed_waymax_outcomes" if row_count > 0 else "initialized_no_valid_outcomes"
            )
        )
        if row_count == 0:
            initialized_no_outcomes += 1
        stored = {_store_key(k): v for k, v in arrays.items()}
        _write_npz_atomic(dst, stored, compress=bool(args.compress))
        attached += int(rollout_valid.any())
        copied += 1
    summary = {
        "processed_or_written": copied,
        "skipped_existing_complete": skipped_existing,
        "rewritten_existing_incomplete": rewritten_existing,
        "skipped_no_outcomes": skipped_no_outcomes,
        "scenes_initialized_without_outcomes": initialized_no_outcomes,
        "scenes_with_attached_valid_outcomes": attached,
        "output_dir": str(out_dir),
        **outcome_stats,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
