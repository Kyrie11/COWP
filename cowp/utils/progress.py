from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

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
    """Wrap an iterable with tqdm when available.

    The production WOMD pipeline can run for hours.  Keeping tqdm optional lets
    unit tests and minimal installs run without a hard runtime dependency, while
    the normal environment still gets progress, throughput and ETA display.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover - tqdm is an optional convenience.
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, **kwargs)
