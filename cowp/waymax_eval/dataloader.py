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


def _maybe_copy_and_update(cfg, **kwargs):
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if not clean:
        return cfg
    try:
        return cfg.copy_and_update(**clean)
    except Exception:
        for k, v in clean.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
        return cfg


def make_default_config(config_name: str = "WOD_1_1_0_TRAINING", path: str | None = None, max_num_objects: int = 128, include_sdc_paths: bool = True):
    _config, _, _ = require_waymax()
    if not hasattr(_config, config_name):
        raise ValueError(f"Unknown Waymax config {config_name}. Available configs are defined in waymax.config.")
    cfg = getattr(_config, config_name)
    cfg = _maybe_copy_and_update(cfg, max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)
    if path is not None:
        cfg = _maybe_copy_and_update(cfg, path=path)
    return cfg


def make_config_from_cowp_config(data_config):
    """Return a real Waymax dataloader config from a COWP yaml dict or a Waymax config.

    The earlier wrapper passed the whole COWP YAML dict directly into
    ``waymax.dataloader.simulator_state_generator``.  That does not actually
    select the WOMD path/config in Waymax.  This adapter keeps the public API
    convenient while ensuring Waymax receives one of its own config objects.
    """
    if not isinstance(data_config, dict):
        return data_config
    womd_cfg = data_config.get("womd", data_config)
    config_name = str(womd_cfg.get("waymax_config_name", "WOD_1_1_0_TRAINING"))
    path = womd_cfg.get("waymax_path") or womd_cfg.get("tfexample_glob") or womd_cfg.get("validation_tfexample_glob")
    max_num_objects = int(womd_cfg.get("max_num_objects", data_config.get("limits", {}).get("max_agents", 128)))
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    return make_default_config(config_name=config_name, path=path, max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)


def waymax_state_generator(data_config):
    _, dataloader, _ = require_waymax()
    cfg = make_config_from_cowp_config(data_config)
    return dataloader.simulator_state_generator(cfg)
