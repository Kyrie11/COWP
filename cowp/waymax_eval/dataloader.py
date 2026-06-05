from __future__ import annotations

from typing import Iterator


def require_waymax():
    try:
        from waymax import config as _config  # type: ignore
        from waymax import dataloader  # type: ignore
        from waymax.dataloader import womd_factories  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Waymax is required for closed-loop rollout. Install waymax and JAX, then configure WOMD paths.") from exc
    return _config, dataloader, womd_factories


def simulator_state_from_womd_dict(example: dict, include_sdc_paths: bool = True, time_key: str = "all"):
    _, _, womd_factories = require_waymax()
    return womd_factories.simulator_state_from_womd_dict(example, include_sdc_paths=include_sdc_paths, time_key=time_key)


def waymax_state_generator(data_config):
    _, dataloader, _ = require_waymax()
    return dataloader.simulator_state_generator(data_config)


def make_default_config(config_name: str = "WOD_1_1_0_TRAINING", path: str | None = None, max_num_objects: int = 128, include_sdc_paths: bool = True):
    _config, _, _ = require_waymax()
    if not hasattr(_config, config_name):
        raise ValueError(f"Unknown Waymax config {config_name}. Available configs are defined in waymax.config.")
    cfg = getattr(_config, config_name)
    try:
        cfg = cfg.copy_and_update(max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)
    except Exception:
        pass
    if path is not None:
        try:
            cfg = cfg.copy_and_update(path=path)
        except Exception:
            cfg.path = path
    return cfg
