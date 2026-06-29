from __future__ import annotations

from typing import Iterator

from cowp.data.parse_tfexample import (
    decode_parsed_tfexample,
    iter_tfexample_records,
    iter_tfexample_records_by_file,
    parse_tfexample,
    scenario_id_from_parsed_tfexample,
)
import dataclasses


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

    if dataclasses.is_dataclass(cfg):
        try:
            return dataclasses.replace(cfg, **clean)
        except TypeError as exc:
            valid = {f.name for f in dataclasses.fields(cfg)}
            bad = sorted(set(clean) - valid)
            raise TypeError(
                f"Invalid Waymax DatasetConfig fields: {bad}. "
                f"Available fields include: {sorted(valid)}"
            ) from exc

    try:
        return cfg.copy_and_update(**clean)
    except Exception:
        for k, v in clean.items():
            setattr(cfg, k, v)
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
        return (
            womd_cfg.get("validation_waymax_path")
            or womd_cfg.get("validation_tfexample_glob")
            or womd_cfg.get("waymax_path")
            or womd_cfg.get("tfexample_glob")
        )
    return (
        womd_cfg.get("waymax_path")
        or womd_cfg.get("tfexample_glob")
        or womd_cfg.get("validation_waymax_path")
        or womd_cfg.get("validation_tfexample_glob")
    )



def _raw_tfexample_path_from_cowp_config(data_config: dict, split: str | None = None) -> str | None:
    """Path pattern usable by TensorFlow TFRecordDataset/id scanning.

    Waymax accepts shard syntax such as ``file@1000``; the lightweight parser in
    cowp.data.parse_tfexample expects normal glob patterns.  Therefore the
    cache-matched replay path should prefer ``tfexample_glob`` fields over
    ``waymax_path`` fields when it scans scenario ids.
    """
    womd_cfg = _womd_subconfig(data_config)
    if split == "validation":
        return womd_cfg.get("validation_tfexample_glob") or womd_cfg.get("tfexample_glob") or womd_cfg.get("validation_waymax_path") or womd_cfg.get("waymax_path")
    return womd_cfg.get("tfexample_glob") or womd_cfg.get("waymax_path") or womd_cfg.get("validation_tfexample_glob") or womd_cfg.get("validation_waymax_path")

def make_config_from_cowp_config(data_config, split: str | None = None, path_override: str | None = None):
    """Return a real Waymax dataloader config from a COWP yaml dict or a Waymax config.

    ``path_override`` is important for candidate replay: the tf.Example stream
    used to obtain scenario ids must be exactly the same stream used by Waymax to
    build SimulatorState objects.  Otherwise the two generators can silently drift
    and attach outcomes to the wrong scenario id.
    """
    if not isinstance(data_config, dict):
        if path_override is None:
            return data_config
        return _maybe_copy_and_update(data_config, path=path_override)
    womd_cfg = _womd_subconfig(data_config)
    config_name = str(womd_cfg.get("waymax_config_name", "WOD_1_1_0_TRAINING"))
    path = path_override or _tfexample_path_from_cowp_config(data_config, split=split)
    max_num_objects = int(womd_cfg.get("max_num_objects", data_config.get("limits", {}).get("max_agents", 128)))
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    return make_default_config(config_name=config_name, path=path, max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)


def waymax_state_generator(data_config, split: str | None = None, tfexample_glob: str | None = None):
    _, dataloader, _ = require_waymax()
    cfg = make_config_from_cowp_config(data_config, split=split, path_override=tfexample_glob)
    return dataloader.simulator_state_generator(config=cfg)


def waymax_state_generator_with_ids(data_config: dict, tfexample_glob: str | None = None, split: str | None = None) -> Iterator[tuple[str, object]]:
    """Yield ``(scenario_id, SimulatorState)`` pairs in Waymax dataloader order.

    This compatibility path still builds states for every record in the split.
    For filtered tensor caches prefer ``waymax_state_generator_for_sids``, which
    scans ids cheaply and materializes Waymax states only for matching scenarios.
    """
    state_path = tfexample_glob or _tfexample_path_from_cowp_config(data_config, split=split)
    id_path = tfexample_glob or _raw_tfexample_path_from_cowp_config(data_config, split=split)
    if state_path is None or id_path is None:
        raise ValueError("A WOMD tf.Example glob/path is required to attach scenario ids to Waymax states.")
    state_iter = waymax_state_generator(data_config, split=split, tfexample_glob=state_path)
    id_iter = iter_tfexample_records(id_path)
    for raw, state in zip(id_iter, state_iter):
        sid = scenario_id_from_parsed_tfexample(parse_tfexample(raw))
        yield sid, state


def waymax_state_generator_for_sids(
    data_config: dict,
    scenario_ids: set[str],
    *,
    tfexample_glob: str | None = None,
    split: str | None = None,
    progress_callback=None,
) -> Iterator[tuple[str, object]]:
    """Yield Waymax states only for requested scenario ids.

    The original replay code zipped Waymax's full-state generator with a second
    tf.Example id generator.  On an interaction-heavy cache this wastes most of
    the runtime because hundreds of thousands of non-cache WOMD scenes are still
    converted into SimulatorState objects.  This generator parses the cheap
    scenario id first; only matching records are decoded and sent through
    ``womd_factories.simulator_state_from_womd_dict``.
    """
    path = tfexample_glob or _raw_tfexample_path_from_cowp_config(data_config, split=split)
    if path is None:
        raise ValueError("A WOMD tf.Example glob/path is required for cache-matched Waymax replay.")
    womd_cfg = _womd_subconfig(data_config)
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    remaining = set(str(x) for x in scenario_ids)
    scanned = 0
    matched = 0
    for raw in iter_tfexample_records(path):
        scanned += 1
        parsed = parse_tfexample(raw)
        sid = scenario_id_from_parsed_tfexample(parsed)
        if sid not in remaining:
            if progress_callback is not None and (scanned == 1 or scanned % 2000 == 0):
                progress_callback(scanned=scanned, matched=matched, remaining=len(remaining), last=sid)
            continue
        example = decode_parsed_tfexample(parsed)
        state = simulator_state_from_womd_dict(example, include_sdc_paths=include_sdc_paths, time_key="all")
        matched += 1
        remaining.remove(sid)
        if progress_callback is not None:
            progress_callback(scanned=scanned, matched=matched, remaining=len(remaining), last=sid)
        yield sid, state
        if not remaining:
            break
