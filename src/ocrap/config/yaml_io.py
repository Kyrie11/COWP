from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .defaults import deep_update, get_default_config


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg = get_default_config()
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
