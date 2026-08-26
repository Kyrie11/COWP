from __future__ import annotations

from typing import Iterator
import json
from pathlib import Path

from cowp.data.parse_tfexample import (
    decode_parsed_tfexample,
    iter_tfexample_records,
    iter_tfexample_records_by_file,
    iter_tfexample_records_sharded,
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


def _has_womd_time_prefix(example: dict, prefix: str) -> bool:
    return any(str(k).startswith(prefix) for k in example.keys())


_WOMD_STATE_TIME_STEPS = {"past": 10, "current": 1, "future": 80}
_WOMD_TL_TIME_STEPS = {"past": 10, "current": 1, "future": 80}


def _as_numpy_array(x):
    try:
        import numpy as _np

        return _np.asarray(x)
    except Exception:
        return x


def _reshape_if_flat(x, shape: tuple[int, ...]):
    """Reshape raw tf.Example list values only when the element count matches."""
    try:
        import numpy as _np

        arr = _np.asarray(x)
        if tuple(arr.shape) == tuple(shape):
            return arr
        if arr.size == int(_np.prod(shape)):
            return arr.reshape(shape)
        return arr
    except Exception:
        return x


def _infer_num_objects(example: dict, default: int = 128) -> int:
    for key in ("state/id", "state/type", "state/is_sdc", "state/tracks_to_predict", "state/objects_of_interest"):
        if key in example:
            try:
                arr = _as_numpy_array(example[key])
                if getattr(arr, "ndim", 0) >= 1 and int(arr.shape[0]) > 0:
                    return int(arr.shape[0])
            except Exception:
                pass
    for prefix, steps in _WOMD_STATE_TIME_STEPS.items():
        for field in ("x", "valid", "bbox_yaw"):
            key = f"state/{prefix}/{field}"
            if key not in example:
                continue
            try:
                arr = _as_numpy_array(example[key])
                if getattr(arr, "ndim", 0) >= 2:
                    return int(arr.shape[0])
                if getattr(arr, "size", 0) and int(arr.size) % int(steps) == 0:
                    return int(arr.size) // int(steps)
            except Exception:
                pass
    return int(default)


def _infer_num_tls(example: dict, default: int = 16) -> int:
    for prefix, steps in _WOMD_TL_TIME_STEPS.items():
        for field in ("x", "valid", "state", "id"):
            key = f"traffic_light_state/{prefix}/{field}"
            if key not in example:
                continue
            try:
                arr = _as_numpy_array(example[key])
                if getattr(arr, "ndim", 0) >= 2:
                    # WOMD traffic-light tensors are [T, num_tls].
                    return int(arr.shape[-1])
                if getattr(arr, "size", 0) and int(arr.size) % int(steps) == 0:
                    return int(arr.size) // int(steps)
            except Exception:
                pass
    return int(default)


def _reshape_womd_arrays_for_waymax(example: dict) -> dict:
    """Restore raw WOMD tf.Example feature-list arrays to Waymax tensor shapes.

    ``decode_parsed_tfexample`` intentionally reads tf.train.Example features
    without TensorFlow's FixedLenFeature schema, so values arrive as flat 1-D
    arrays.  Waymax factories expect shaped tensors, e.g. state [N,T],
    traffic lights [T,L], roadgraph xyz [P,3], and route paths [R,Q,3].
    """
    ex = dict(example)
    n_obj = _infer_num_objects(ex)

    # Agent temporal tensors: [num_objects, num_steps].
    for prefix, steps in _WOMD_STATE_TIME_STEPS.items():
        base = f"state/{prefix}/"
        for key in list(ex.keys()):
            if key.startswith(base):
                ex[key] = _reshape_if_flat(ex[key], (n_obj, int(steps)))

    # Roadgraph tensors.  Waymax indexes type/id/valid with [..., 0].
    for key in ("roadgraph_samples/xyz", "roadgraph_samples/dir"):
        if key in ex:
            try:
                arr = _as_numpy_array(ex[key])
                if getattr(arr, "ndim", 0) == 1 and int(arr.size) % 3 == 0:
                    ex[key] = arr.reshape(-1, 3)
            except Exception:
                pass
    for key in ("roadgraph_samples/id", "roadgraph_samples/type", "roadgraph_samples/valid"):
        if key in ex:
            try:
                arr = _as_numpy_array(ex[key])
                if getattr(arr, "ndim", 0) == 1:
                    ex[key] = arr.reshape(-1, 1)
            except Exception:
                pass

    # SDC route paths introduced in WOMD 1.3.1.
    if "path_samples/xyz" not in ex and all(k in ex for k in ("path_samples/x", "path_samples/y", "path_samples/z")):
        try:
            import numpy as _np

            ex["path_samples/xyz"] = _np.stack(
                [_as_numpy_array(ex["path_samples/x"]), _as_numpy_array(ex["path_samples/y"]), _as_numpy_array(ex["path_samples/z"])],
                axis=-1,
            )
        except Exception:
            pass
    if "path_samples/xyz" in ex:
        try:
            arr = _as_numpy_array(ex["path_samples/xyz"])
            if getattr(arr, "ndim", 0) == 1 and int(arr.size) % 3 == 0:
                num_paths = 45
                on_route = _as_numpy_array(ex.get("path_samples/on_route", []))
                if getattr(on_route, "size", 0) > 0:
                    num_paths = int(on_route.size)
                num_points = max(int(arr.size) // (int(num_paths) * 3), 1)
                if int(arr.size) == int(num_paths) * int(num_points) * 3:
                    ex["path_samples/xyz"] = arr.reshape(int(num_paths), int(num_points), 3)
        except Exception:
            pass
    if "path_samples/xyz" in ex:
        try:
            xyz = _as_numpy_array(ex["path_samples/xyz"])
            if getattr(xyz, "ndim", 0) >= 3:
                num_paths, num_points = int(xyz.shape[0]), int(xyz.shape[1])
                for key in ("path_samples/id", "path_samples/valid", "path_samples/arc_length"):
                    if key in ex:
                        ex[key] = _reshape_if_flat(ex[key], (num_paths, num_points))
                if "path_samples/on_route" in ex:
                    ex["path_samples/on_route"] = _reshape_if_flat(ex["path_samples/on_route"], (num_paths, 1))
        except Exception:
            pass

    # Traffic-light tensors: non-timestamp fields are [num_steps, num_tls],
    # timestamp_micros is [num_steps].
    n_tls = _infer_num_tls(ex)
    for prefix, steps in _WOMD_TL_TIME_STEPS.items():
        base = f"traffic_light_state/{prefix}/"
        for key in list(ex.keys()):
            if not key.startswith(base):
                continue
            field = key[len(base) :]
            if field == "timestamp_micros":
                ex[key] = _reshape_if_flat(ex[key], (int(steps),))
            else:
                ex[key] = _reshape_if_flat(ex[key], (int(steps), int(n_tls)))
    return ex


def _aggregate_womd_time_tensors_numpy(example: dict) -> dict:
    """Add state/all/* and traffic_light_state/all/* using NumPy arrays."""
    ex = dict(example)
    try:
        import numpy as _np
    except Exception:
        return ex

    state_features = set()
    for key in ex:
        if key.startswith("state/current/"):
            state_features.add(key[len("state/current/") :])
    for feat in state_features:
        out_key = f"state/all/{feat}"
        if out_key in ex:
            continue
        parts = [f"state/{part}/{feat}" for part in ("past", "current", "future")]
        if all(k in ex for k in parts):
            try:
                ex[out_key] = _np.concatenate([_as_numpy_array(ex[k]) for k in parts], axis=-1)
            except Exception:
                pass

    tl_features = set()
    for key in ex:
        if key.startswith("traffic_light_state/current/"):
            tl_features.add(key[len("traffic_light_state/current/") :])
    for feat in tl_features:
        out_key = f"traffic_light_state/all/{feat}"
        if out_key in ex:
            continue
        parts = [f"traffic_light_state/{part}/{feat}" for part in ("past", "current", "future")]
        if all(k in ex for k in parts):
            try:
                # After reshaping, both timestamp [T] and signal fields [T,L]
                # concatenate over the timestep axis 0.
                ex[out_key] = _np.concatenate([_as_numpy_array(ex[k]) for k in parts], axis=0)
            except Exception:
                pass

    if "state/which_time" not in ex:
        try:
            past_len = int(_as_numpy_array(ex["state/past/valid"]).shape[-1])
            future_len = int(_as_numpy_array(ex["state/future/valid"]).shape[-1])
            ex["state/which_time"] = _np.concatenate(
                [-_np.ones((past_len,), dtype=_np.float32), _np.zeros((1,), dtype=_np.float32), _np.ones((future_len,), dtype=_np.float32)],
                axis=0,
            )
        except Exception:
            pass
    return ex


def _maybe_aggregate_time_tensors(example: dict, *, time_key: str = "all") -> dict:
    """Return a WOMD dict accepted by Waymax's simulator-state factory.

    The cache-matched replay path first filters scenario ids cheaply from raw
    serialized tf.Examples and therefore bypasses Waymax's TensorFlow parser.
    Raw feature lists must be reshaped and ``past/current/future`` tensors must
    be merged under ``all`` before calling Waymax factories.
    """
    ex = _reshape_womd_arrays_for_waymax(dict(example))
    if str(time_key) != "all":
        return ex
    has_state_all = _has_womd_time_prefix(ex, "state/all/")
    has_tl_split = _has_womd_time_prefix(ex, "traffic_light_state/current/")
    has_tl_all = _has_womd_time_prefix(ex, "traffic_light_state/all/")
    if has_state_all and (not has_tl_split or has_tl_all):
        return ex
    return _aggregate_womd_time_tensors_numpy(ex)


def _has_nonempty_key(example: dict, key: str) -> bool:
    if key not in example:
        return False
    try:
        import numpy as _np

        return _np.asarray(example[key]).size > 0
    except Exception:
        return True


def _has_required_sdc_path_samples(example: dict) -> bool:
    """Return whether the dict contains usable route-path features for Waymax.

    ``include_sdc_paths=True`` asks Waymax to construct ``datatypes.Paths`` from
    ``path_samples/*``.  Some WOMD tf.Example releases/splits do not carry these
    route-path features.  In that case Waymax's factory may fall back to scalar
    placeholders and fail during ``Paths.validate()`` before safety replay even
    starts.  Collision/offroad candidate labels do not require routes, so the
    lightweight cache-matched replay path should only request SDC paths when the
    required route tensors are actually present.
    """
    # Public Waymax/WOMD 1.3.1 route-path features are stored as path_samples/xyz
    # plus id/valid/arc_length/on_route.  Some older local caches used x/y/z, so
    # accept either coordinate layout and synthesize xyz in the reshaping helper.
    has_xyz = _has_nonempty_key(example, "path_samples/xyz") or all(
        _has_nonempty_key(example, key) for key in ("path_samples/x", "path_samples/y", "path_samples/z")
    )
    required = ("path_samples/id", "path_samples/valid", "path_samples/arc_length", "path_samples/on_route")
    return bool(has_xyz) and all(_has_nonempty_key(example, key) for key in required)


def simulator_state_from_womd_dict(example: dict, include_sdc_paths: bool = True, time_key: str = "all"):
    _, _, womd_factories = require_waymax()
    example_for_waymax = _maybe_aggregate_time_tensors(example, time_key=time_key)
    include_paths = bool(include_sdc_paths) and _has_required_sdc_path_samples(example_for_waymax)
    return womd_factories.simulator_state_from_womd_dict(example_for_waymax, include_sdc_paths=include_paths, time_key=time_key)




def womd_example_from_tensor_cache_arrays(arrays: dict) -> dict:
    """Recover a WOMD tf.Example-style dict from a COWP tensor-cache item.

    ``02_build_tensor_cache`` stores raw WOMD tf.Example features under the
    ``womd/`` namespace with slash-safe NPZ keys. Candidate replay can build
    Waymax SimulatorState directly from these cached features instead of
    scanning the original TFRecords again. This is a safe optimization: it uses
    the exact WOMD features that were merged into the cache for the same scene.
    """
    import numpy as _np

    ex: dict = {}
    for key, value in dict(arrays).items():
        k = str(key).replace("__", "/")
        if not k.startswith("womd/"):
            continue
        out_key = k[len("womd/") :]
        arr = _np.asarray(value)
        if arr.dtype == _np.uint8 and out_key in {"scenario/id", "scenario/id_bytes"}:
            try:
                ex[out_key] = bytes(arr.tolist())
                continue
            except Exception:
                pass
        ex[out_key] = arr
    if not ex:
        raise KeyError("tensor cache item does not contain womd/* features; rebuild tensor cache with 02_build_tensor_cache before cache-source Waymax replay")
    return ex


def simulator_state_from_tensor_cache_arrays(arrays: dict, data_config: dict | None = None, *, include_sdc_paths: bool | None = None, time_key: str = "all"):
    """Build a Waymax SimulatorState from cached WOMD arrays."""
    if include_sdc_paths is None:
        womd_cfg = _womd_subconfig(data_config or {}) if isinstance(data_config, dict) else {}
        include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    example = womd_example_from_tensor_cache_arrays(arrays)
    return simulator_state_from_womd_dict(example, include_sdc_paths=bool(include_sdc_paths), time_key=time_key)

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
    # Make the Waymax/WOMD shape contract explicit instead of relying on named-
    # config defaults that may change across Waymax releases.  SimulatorState
    # construction requires a scenario-level, time-aggregated example.
    cfg = _maybe_copy_and_update(
        cfg,
        max_num_objects=max_num_objects,
        include_sdc_paths=include_sdc_paths,
        aggregate_timesteps=True,
        batch_by_scenario=True,
    )
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

    ``cowp.data.parse_tfexample`` accepts both normal globs and Waymax shard
    syntax such as ``file@1000``.  Prefer explicit raw tf.Example fields because
    they also work for provenance/index based scans.
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
    # Keep the named Waymax split consistent with the raw stream.  WOMD 1.3.1
    # TRAINING/VALIDATION currently share the same tensor-shape contract, but
    # selecting the proper named config is safer across Waymax releases and makes
    # the closed-loop provenance unambiguous in experiment logs.
    if split == "validation":
        explicit_validation = womd_cfg.get("validation_waymax_config_name")
        if explicit_validation:
            config_name = str(explicit_validation)
        elif config_name.endswith("_TRAINING"):
            candidate = config_name[: -len("_TRAINING")] + "_VALIDATION"
            try:
                _config, _, _ = require_waymax()
                if hasattr(_config, candidate):
                    config_name = candidate
            except Exception:
                # require_waymax() will raise a clearer error later when the
                # dataloader is actually instantiated.
                pass
    path = path_override or _tfexample_path_from_cowp_config(data_config, split=split)
    max_num_objects = int(womd_cfg.get("max_num_objects", data_config.get("limits", {}).get("max_agents", 128)))
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    return make_default_config(config_name=config_name, path=path, max_num_objects=max_num_objects, include_sdc_paths=include_sdc_paths)


def waymax_state_generator(data_config, split: str | None = None, tfexample_glob: str | None = None):
    _, dataloader, _ = require_waymax()
    cfg = make_config_from_cowp_config(data_config, split=split, path_override=tfexample_glob)
    return dataloader.simulator_state_generator(config=cfg)


def waymax_state_generator_sharded(
    data_config: dict,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    tfexample_glob: str | None = None,
    split: str | None = None,
) -> Iterator[tuple[int, object]]:
    """Yield only this process's ``(global_record_index, SimulatorState)`` rows.

    The previous online evaluator enumerated Waymax's full state generator and
    applied ``raw_index % num_shards`` *after* each SimulatorState had already
    been parsed and materialized.  With N workers that repeats the most expensive
    input conversion N times.  This generator applies the exact same modulo
    assignment to serialized TFExample records first, and constructs a Waymax
    state only for records owned by the current process.

    Record ordering and the modulo rule are unchanged.  Therefore scenario
    membership, scenario_index values, policy calls, environment stepping, and
    metrics are identical to the original sharded evaluation path.
    """
    num_shards = max(int(num_shards), 1)
    shard_index = int(shard_index) % num_shards
    if num_shards == 1:
        for raw_index, state in enumerate(waymax_state_generator(data_config, split=split, tfexample_glob=tfexample_glob)):
            yield raw_index, state
        return

    path = tfexample_glob or _raw_tfexample_path_from_cowp_config(data_config, split=split)
    if path is None:
        raise ValueError("A WOMD tf.Example glob/path is required for sharded Waymax rollout.")
    womd_cfg = _womd_subconfig(data_config)
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    for raw_index, raw in iter_tfexample_records_sharded(
        path,
        shard_index=shard_index,
        num_shards=num_shards,
    ):
        parsed = parse_tfexample(raw)
        example = decode_parsed_tfexample(parsed)
        state = simulator_state_from_womd_dict(example, include_sdc_paths=include_sdc_paths, time_key="all")
        yield raw_index, state


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



def _files_for_sids_from_tfexample_index(index_jsonl: str | Path, scenario_ids: set[str]) -> tuple[list[str], dict[str, object]]:
    """Return TFRecord shard files containing requested scenario ids.

    The tensor-cache build stage can already create a lightweight JSONL index with
    {scenario_id, file, record_index}.  Candidate replay cannot random-seek into
    compressed TFRecords portably, but it can avoid scanning shards that do not
    contain any cache scene.  This is a safe I/O optimization: it changes only
    which TFRecord files are streamed, not how SimulatorState objects are built.
    """
    p = Path(index_jsonl)
    files: set[str] = set()
    matched: set[str] = set()
    rows = 0
    if not p.exists():
        return [], {"index": str(p), "exists": False, "indexed_rows": 0, "matched_sids": 0, "indexed_files": 0}
    targets = set(str(x) for x in scenario_ids)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = row.get("scenario_id")
            fpath = row.get("file")
            if sid in targets and fpath:
                matched.add(str(sid))
                files.add(str(fpath))
    return sorted(files), {
        "index": str(p),
        "exists": True,
        "indexed_rows": rows,
        "matched_sids": len(matched),
        "indexed_files": len(files),
    }

def waymax_state_generator_for_sids(
    data_config: dict,
    scenario_ids: set[str],
    *,
    tfexample_glob: str | None = None,
    split: str | None = None,
    tfexample_index_jsonl: str | Path | None = None,
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
    index_stats = None
    if tfexample_index_jsonl is not None:
        indexed_files, index_stats = _files_for_sids_from_tfexample_index(tfexample_index_jsonl, set(str(x) for x in scenario_ids))
        if indexed_files:
            path = indexed_files
    womd_cfg = _womd_subconfig(data_config)
    include_sdc_paths = bool(womd_cfg.get("include_sdc_paths", True))
    remaining = set(str(x) for x in scenario_ids)
    scanned = 0
    matched = 0
    if progress_callback is not None and index_stats is not None:
        progress_callback(scanned=0, matched=0, remaining=len(remaining), last=f"index_files={index_stats.get('indexed_files', 0)}")
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
