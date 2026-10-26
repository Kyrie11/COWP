from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_update(cfg, load_yaml(path))
    cfg = deep_update(cfg, overrides)
    return cfg


def get_cfg(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require_cfg(cfg: Mapping[str, Any], dotted: str) -> Any:
    value = get_cfg(cfg, dotted, None)
    if value is None:
        raise KeyError(f"Missing required config key: {dotted}")
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
