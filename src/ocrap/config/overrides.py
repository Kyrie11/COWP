from __future__ import annotations

from typing import Any

import yaml

from .defaults import deep_update


def _parse_scalar(s: str) -> Any:
    # ``--set dotted.path=`` is used throughout the shell launchers to mean an
    # explicit empty string (for example scratch ``init_checkpoint`` and
    # optional component-reliability CSVs).  ``yaml.safe_load("")`` returns
    # ``None``, which silently changed that contract and later produced the
    # literal string ``"None"`` in string-valued model settings.  Preserve an
    # explicitly empty CLI value; callers can still request YAML null with
    # ``key=null`` or ``key=~``.
    if s == "":
        return ""
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def parse_cli_overrides(items: list[str] | None) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        cur = root
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _parse_scalar(value)
    return root


def apply_overrides(cfg: dict[str, Any], items: list[str] | None) -> dict[str, Any]:
    return deep_update(cfg, parse_cli_overrides(items))
