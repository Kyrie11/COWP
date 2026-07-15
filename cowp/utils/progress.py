from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar
import os
import sys

T = TypeVar("T")


def tqdm_iter(
    iterable: Iterable[T],
    *,
    enabled: bool = True,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
    **kwargs,
) -> Iterable[T]:
    """Wrap an iterable with tqdm when available and make it visible in logs.

    WOMD label construction can spend a long time inside one scenario before the
    outer loop advances.  The wrapper therefore forces a standard stderr tqdm
    instance, low mininterval, and an initial refresh so that SSH sessions and
    redirected logs show that the job is alive.
    """
    if not enabled or os.environ.get("COWP_NO_PROGRESS", "0") in {"1", "true", "True"}:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - tqdm is an optional convenience.
        return iterable
    defaults = {
        "dynamic_ncols": True,
        "file": sys.stderr,
        "mininterval": 1.0,
        "maxinterval": 5.0,
        "miniters": 1,
        "leave": True,
        "ascii": not bool(getattr(sys.stderr, "encoding", "") and "UTF" in str(sys.stderr.encoding).upper()),
    }
    defaults.update(kwargs)
    tqdm_kwargs = dict(defaults)
    if total is not None:
        tqdm_kwargs["total"] = total
    if desc is not None:
        tqdm_kwargs["desc"] = desc
    if unit is not None:
        tqdm_kwargs["unit"] = unit
    bar = tqdm(iterable, **tqdm_kwargs)
    try:
        bar.refresh()
    except Exception:
        pass
    return bar
