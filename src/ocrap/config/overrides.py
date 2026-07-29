from __future__ import annotations

from typing import Any

import yaml

from .defaults import deep_update


def _parse_scalar(s: str) -> Any:
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
