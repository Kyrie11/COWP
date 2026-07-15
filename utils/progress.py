from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar
import os
import sys

T = TypeVar("T")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def tqdm_iter(
    iterable: Iterable[T],
    *,
    enabled: bool = True,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
    **kwargs,
) -> Iterable[T]:
    """Wrap an iterable with a visible tqdm bar.

    The external baseline scripts are usually launched from shell wrappers that may
    tee stdout/stderr into log files.  This wrapper keeps tqdm enabled unless the
    caller explicitly disables it, uses a deterministic bar format, and supports
    per-process bar positions through ``COWP_TQDM_POSITION`` so GameFormer and
    DTPP can train concurrently on two GPUs without completely overwriting each
    other's epoch bars.
    """
    if not enabled or _env_bool("COWP_NO_PROGRESS", False):
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - tqdm is an optional convenience.
        return iterable

    position_env = os.environ.get("COWP_TQDM_POSITION")
    position = None
    if position_env not in (None, ""):
        try:
            position = int(position_env)
        except ValueError:
            position = None

    defaults = {
        "dynamic_ncols": True,
        "file": sys.stderr,
        "mininterval": 0.5,
        "maxinterval": 2.0,
        "miniters": 1,
        "leave": True,
        "disable": False,
        "smoothing": 0.1,
        "ascii": not bool(getattr(sys.stderr, "encoding", "") and "UTF" in str(sys.stderr.encoding).upper()),
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }
    if position is not None:
        defaults["position"] = position
    defaults.update(kwargs)
    if total is not None:
        defaults["total"] = total
    if desc is not None:
        defaults["desc"] = desc
    if unit is not None:
        defaults["unit"] = unit

    bar = tqdm(iterable, **defaults)
    try:
        bar.refresh()
    except Exception:
        pass
    return bar
