from __future__ import annotations

from typing import Iterator

from cowp.data.parse_tfexample import iter_tfexample_records, parse_tfexample, scenario_id_from_parsed_tfexample


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


def _womd_subconfig(data_config: dict) -> dict:
    return data_config.get("womd", data_config)


def _tfexample_path_from_cowp_config(data_config: dict, split: str | None = None) -> str | None:
    womd_cfg = _womd_subconfig(data_config)
    if split == "validation":
        return womd_cfg.get("validation_tfexample_glob") or womd_cfg.get("waymax_path") or womd_cfg.get("tfexample_glob")
    return womd_cfg.get("waymax_path") or womd_cfg.get("tfexample_glob") or womd_cfg.get("validation_tfexample_glob")


def make_config_from_cowp_config(data_config, split: str | None = None):
    """Return a real Waymax dataloader config from a COWP yaml dict or a Waymax config.

    The adapter ensures Waymax receives one of its own config objects and a WOMD
    tf.Example path, rather than the whole COWP YAML dict.
    """
    if not isinstance(data_config, dict):
        return data_config
    womd_cfg = _womd_subconfig(data_config)
    config_name = str(womd_cfg.get("waymax_config_name", "WOD_1_1_0_TRAINING"))
    path = _tfexample_path_from_cowp_config(data_config, split=split)
    max_num_objects = int(womd_cfg.get("max_num_objects", data_config.get("limits", {}).get("max_agents", 128)))
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    return make_default_config(config_name=config_name, path=path, max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)


def waymax_state_generator(data_config, split: str | None = None):
    _, dataloader, _ = require_waymax()
    cfg = make_config_from_cowp_config(data_config, split=split)
    return dataloader.simulator_state_generator(config=cfg)


def waymax_state_generator_with_ids(data_config: dict, tfexample_glob: str | None = None, split: str | None = None) -> Iterator[tuple[str, object]]:
    """Yield ``(scenario_id, SimulatorState)`` pairs.

    Waymax's SimulatorState API does not expose a scenario-id field in the public
    dataclass.  We therefore parse the paired WOMD tf.Example stream in the same
    order and attach its ``scenario/id`` to the generated state.  Use the same
    tf.Example glob/path for both streams.
    """
    path = tfexample_glob or _tfexample_path_from_cowp_config(data_config, split=split)
    if path is None:
        raise ValueError("A WOMD tf.Example glob/path is required to attach scenario ids to Waymax states.")
    state_iter = waymax_state_generator(data_config, split=split)
    id_iter = iter_tfexample_records(path)
    for raw, state in zip(id_iter, state_iter):
        sid = scenario_id_from_parsed_tfexample(parse_tfexample(raw))
        yield sid, state
