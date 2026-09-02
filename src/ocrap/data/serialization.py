from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: dict[str, Any], path: str | Path, *, fsync: bool = False) -> None:
    """Atomically write JSON so an interrupted process never leaves a torn file.

    Closed-loop evaluation updates progress/partial files while a long Waymax run
    is active.  Writing directly to the destination could leave invalid JSON when
    the process is killed during ``json.dump``.  A same-directory temporary file
    plus ``os.replace`` preserves the previous valid snapshot until the new one is
    complete.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = (p.stat().st_mode & 0o777) if p.exists() else 0o644
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def np_savez(path: str | Path, *, compressed: bool = True, fsync: bool = True, **arrays: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            # Passing an open file handle prevents NumPy from silently appending
            # an extra .npz suffix to our temporary file name.
            saver = np.savez_compressed if compressed else np.savez
            saver(f, **arrays)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_npz_selected(path: str | Path, keys: set[str] | frozenset[str] | tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Load only selected members from an NPZ archive.

    ``np.savez_compressed`` stores every array as an independent ZIP member.  The
    historical training/calibration path eagerly materialized *all* members even
    though the model consumes only a stable subset.  Reading just the requested
    members is numerically identical for those keys and avoids decompression and
    allocation of unused rollout/debug tensors.  Missing optional keys are left
    absent so existing ``dict.get`` fallbacks retain their exact semantics.
    """
    wanted = frozenset(str(k) for k in keys)
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files if k in wanted}


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def parse_json_field(x: Any, default: Any) -> Any:
    try:
        s = scalar_str(x)
        return json.loads(s)
    except Exception:
        return default
