from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait, as_completed
from pathlib import Path
from typing import Mapping

import numpy as np

from cowp.data.parse_scenario_proto import iter_scenario_records, iter_scenarios, scenario_to_scene
from cowp.data.parse_tfexample import decode_parsed_tfexample, iter_tfexample_records, iter_tfexample_records_by_file, parse_tfexample, resolve_glob_patterns, scenario_id_from_parsed_tfexample
from cowp.geometry.lane_graph import build_conflict_regions
from cowp.label.label_engine import build_labels_for_scene
from cowp.label.scene_filter import is_interaction_heavy, valid_scene_basic
from cowp.utils.progress import tqdm_iter
from cowp.data.dataset import align_critical_agents_to_womd_input, mask_out_of_range_critical_agents


_LABEL_WORKER_STATE: dict[str, object] = {}
_CACHE_WORKER_STATE: dict[str, object] = {}


def _npz_key(key: str) -> str:
    return key.replace("/", "__")


def _write_npz(path: str | Path, arrays: Mapping[str, object], *, compress: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _label_npz_looks_complete(path: str | Path, expected_sid: str) -> tuple[bool, str]:
    """Cheap integrity check used by --skip-existing.

    A previous interrupted run can leave a zero-byte/partial/corrupt file, and a
    stale file can also have the wrong scenario id.  Skipping on path existence
    alone can silently poison resume runs, so we verify a few mandatory keys and
    load their headers before declaring the file reusable.
    """
    path = Path(path)
    if not path.exists():
        return False, "missing"
    required = (
        "scenario/id",
        "cowp/candidates/trajectory",
        "cowp/candidates/valid",
        "cowp/critical/track_index",
        "cowp/natural/traj",
        "cowp/response/traj",
        "cowp/witness/exists",
    )
    try:
        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            missing = [k for k in required if k not in files]
            if missing:
                return False, "missing_keys:" + ",".join(missing[:3])
            sid_arr = data["scenario/id"]
            sid = str(sid_arr.item() if getattr(sid_arr, "shape", ()) == () else sid_arr)
            if sid != expected_sid:
                return False, f"scenario_id_mismatch:{sid}"
            # Force header/data access on small mandatory arrays.  np.load is lazy,
            # so this catches corrupt central-directory or truncated members while
            # avoiding a full read of all large tensors.
            _ = data["cowp/candidates/valid"].shape
            _ = data["cowp/witness/exists"].shape
    except Exception as exc:
        return False, f"load_error:{type(exc).__name__}"
    return True, "ok"


def _init_label_worker(
    output_dir: str,
    cfg: dict,
    ablation: dict | None,
    require_interaction_heavy: bool,
    collect_scene_metadata: bool,
    skip_existing: bool,
    compress: bool,
    allow_scenario_ids: set[str] | None,
    exclude_scenario_ids: set[str] | None,
    profile_label_engine: bool,
) -> None:
    _LABEL_WORKER_STATE.clear()
    _LABEL_WORKER_STATE.update(
        {
            "output_dir": output_dir,
            "cfg": cfg,
            "ablation": ablation,
            "require_interaction_heavy": require_interaction_heavy,
            "collect_scene_metadata": collect_scene_metadata,
            "skip_existing": skip_existing,
            "compress": compress,
            "allow_scenario_ids": allow_scenario_ids,
            "exclude_scenario_ids": exclude_scenario_ids,
            "profile_label_engine": profile_label_engine,
        }
    )


def _build_one_label_worker_task(raw: bytes) -> dict[str, object]:
    if not _LABEL_WORKER_STATE:
        raise RuntimeError("Label worker state was not initialized.")
    return _build_one_label_from_raw(
        raw,
        str(_LABEL_WORKER_STATE["output_dir"]),
        _LABEL_WORKER_STATE["cfg"],
        _LABEL_WORKER_STATE["ablation"],
        bool(_LABEL_WORKER_STATE["require_interaction_heavy"]),
        bool(_LABEL_WORKER_STATE["collect_scene_metadata"]),
        bool(_LABEL_WORKER_STATE["skip_existing"]),
        bool(_LABEL_WORKER_STATE["compress"]),
        _LABEL_WORKER_STATE["allow_scenario_ids"],
        _LABEL_WORKER_STATE["exclude_scenario_ids"],
        bool(_LABEL_WORKER_STATE.get("profile_label_engine", False)),
    )


def save_label_npz(label: Mapping[str, object], path: str | Path, *, compress: bool = True) -> None:
    arrays = {}
    for k, v in label.items():
        if isinstance(v, str):
            arrays[k] = np.asarray(v)
        else:
            arrays[k] = np.asarray(v)
    _write_npz(path, arrays, compress=compress)


def _import_scenario_proto():
    from cowp.data.parse_scenario_proto import _import_scenario_proto as _loader

    return _loader()


def _build_one_label_from_raw(
    raw: bytes,
    output_dir: str,
    cfg: dict,
    ablation: dict | None,
    require_interaction_heavy: bool,
    collect_scene_metadata: bool,
    skip_existing: bool,
    compress: bool,
    allow_scenario_ids: set[str] | None = None,
    exclude_scenario_ids: set[str] | None = None,
    profile_label_engine: bool = False,
) -> dict[str, object]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    scenario_pb2 = _import_scenario_proto()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw)
    sid = str(scenario.scenario_id)
    timings["parse_proto_s"] = time.perf_counter() - t

    if allow_scenario_ids is not None and sid not in allow_scenario_ids:
        return {"status": "filtered", "scenario_id": sid, "filter_reason": "not_in_allow_scenario_ids", "seconds": time.perf_counter() - t0, "timings": timings}
    if exclude_scenario_ids is not None and sid in exclude_scenario_ids:
        return {"status": "filtered", "scenario_id": sid, "filter_reason": "in_exclude_scenario_ids", "seconds": time.perf_counter() - t0, "timings": timings}

    label_path = Path(output_dir) / f"{sid}.npz"
    if skip_existing and label_path.exists():
        t = time.perf_counter()
        ok, reason = _label_npz_looks_complete(label_path, sid)
        timings["skip_existing_check_s"] = time.perf_counter() - t
        if ok:
            return {"status": "existing", "scenario_id": sid, "seconds": time.perf_counter() - t0, "timings": timings}
        timings["skip_existing_rebuild_reason"] = reason

    t = time.perf_counter()
    scene = scenario_to_scene(scenario, keep_raw=False)
    timings["scenario_to_scene_s"] = time.perf_counter() - t

    regions = None
    heavy_meta: dict[str, object] | None = None
    filter_reason = ""
    if require_interaction_heavy or collect_scene_metadata:
        # Cheap gate first.  Some invalid WOMD scenarios should not pay the
        # O(lane-pairs * segment-pairs) conflict-region cost.
        if require_interaction_heavy:
            t = time.perf_counter()
            basic_ok, basic_reasons = valid_scene_basic(scene, cfg)
            timings["basic_filter_s"] = time.perf_counter() - t
            if not basic_ok:
                return {
                    "status": "filtered",
                    "scenario_id": sid,
                    "filter_reason": ",".join(basic_reasons),
                    "seconds": time.perf_counter() - t0,
                    "timings": timings,
                }

        t = time.perf_counter()
        regions = build_conflict_regions(scene.map_data, cfg)
        timings["conflict_regions_s"] = time.perf_counter() - t

        t = time.perf_counter()
        heavy, meta = is_interaction_heavy(scene, cfg, conflict_regions=regions)
        timings["interaction_filter_s"] = time.perf_counter() - t
        heavy_meta = dict(meta)
        heavy_meta["interaction_heavy"] = bool(heavy)
        if require_interaction_heavy and not heavy:
            filter_reason = "not_interaction_heavy"
            return {
                "status": "filtered",
                "scenario_id": sid,
                "filter_reason": filter_reason,
                "num_conflict_regions": len(regions),
                "seconds": time.perf_counter() - t0,
                "timings": timings,
            }

    t = time.perf_counter()
    engine_timings: dict[str, float] | None = {} if profile_label_engine else None
    label = build_labels_for_scene(scene, cfg, ablation=ablation, scene_meta=heavy_meta, conflict_regions=regions, profile_timings=engine_timings)
    if engine_timings:
        timings.update(engine_timings)
    timings["label_engine_s"] = time.perf_counter() - t

    t = time.perf_counter()
    save_label_npz(label, label_path, compress=compress)
    timings["write_npz_s"] = time.perf_counter() - t
    return {
        "status": "written",
        "scenario_id": sid,
        "path": str(label_path),
        "num_conflict_regions": len(regions) if regions is not None else 0,
        "seconds": time.perf_counter() - t0,
        "timings": timings,
    }


def _append_profile(profile_jsonl: str | Path | None, result: Mapping[str, object]) -> None:
    if not profile_jsonl:
        return
    p = Path(profile_jsonl)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(result), ensure_ascii=False) + "\n")


def _count_jsonl_rows(path: str | Path | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        return sum(1 for line in f if line.strip())


def _advance_progress(iterator) -> None:
    """Advance either a tqdm object or a plain iterator by one processed item."""
    if hasattr(iterator, "update"):
        try:
            iterator.update(1)
            return
        except Exception:
            pass
    try:
        next(iterator)
    except Exception:
        pass


def build_labels_from_proto(
    proto_glob: str | list[str],
    output_dir: str | Path,
    cfg: dict,
    limit: int | None = None,
    ablation: dict | None = None,
    progress: bool = True,
    skip_existing: bool = False,
    num_workers: int = 1,
    max_scenarios_scanned: int | None = None,
    compress: bool = True,
    profile_jsonl: str | Path | None = None,
    index_jsonl: str | Path | None = None,
    start_method: str | None = None,
    max_pending_multiplier: int = 4,
    fail_on_error: bool = True,
    allow_scenario_ids: set[str] | None = None,
    exclude_scenario_ids: set[str] | None = None,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if profile_jsonl:
        Path(profile_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(profile_jsonl).write_text("", encoding="utf-8")

    count = 0
    scanned = 0
    skipped_filter = 0
    skipped_existing = 0
    errors = 0
    processed = 0
    filter_cfg = cfg.get("dataset", {})
    require_interaction_heavy = bool(filter_cfg.get("require_interaction_heavy", False))
    collect_scene_metadata = bool(filter_cfg.get("collect_scene_metadata", True))
    profile_label_engine = bool(profile_jsonl)
    total = max_scenarios_scanned
    if total is None:
        total = _count_jsonl_rows(index_jsonl)
    desc = "Build COWP labels from Scenario protos"

    def handle_result(res: Mapping[str, object], iterator) -> None:
        nonlocal count, skipped_filter, skipped_existing, errors, processed
        processed += 1
        status = str(res.get("status", "unknown"))
        if status == "written":
            count += 1
        elif status == "filtered":
            skipped_filter += 1
        elif status == "existing":
            skipped_existing += 1
        elif status == "error":
            errors += 1
        _append_profile(profile_jsonl, res)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                submitted=scanned,
                processed=processed,
                written=count,
                filtered=skipped_filter,
                existing=skipped_existing,
                errors=errors,
                status=status,
                last_s=f"{float(res.get('seconds', 0.0)):.1f}",
                last=str(res.get("scenario_id", ""))[:10],
                refresh=True,
            )

    if int(num_workers) <= 1:
        iterator = tqdm_iter(iter_scenario_records(proto_glob), enabled=progress, total=total, desc=desc, unit="scenario")
        for raw in iterator:
            scanned += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(scanned=scanned, written=count, filtered=skipped_filter, existing=skipped_existing, stage="processing", refresh=True)
            res = _build_one_label_from_raw(
                raw,
                str(output_dir),
                cfg,
                ablation,
                require_interaction_heavy,
                collect_scene_metadata,
                skip_existing,
                compress,
                allow_scenario_ids,
                exclude_scenario_ids,
                profile_label_engine,
            )
            handle_result(res, iterator)
            if limit is not None and count >= limit:
                break
            if max_scenarios_scanned is not None and scanned >= max_scenarios_scanned:
                break
        return count

    workers = max(1, int(num_workers))
    max_pending = max(workers * max(1, int(max_pending_multiplier)), workers)
    raw_iter = iter(iter_scenario_records(proto_glob))
    futures = set()
    stop_submit = False
    progress_source = range(total) if total is not None else iter(int, 1)
    iterator = tqdm_iter(progress_source, enabled=progress, total=total, desc=f"{desc} ({workers} workers)", unit="scenario")

    def submit_one(pool: ProcessPoolExecutor) -> bool:
        nonlocal scanned, stop_submit
        if stop_submit:
            return False
        if max_scenarios_scanned is not None and scanned >= max_scenarios_scanned:
            stop_submit = True
            return False
        try:
            raw = next(raw_iter)
        except StopIteration:
            stop_submit = True
            return False
        scanned += 1
        fut = pool.submit(
            _build_one_label_worker_task,
            raw,
        )
        futures.add(fut)
        return True

    mp_context = mp.get_context(start_method) if start_method else None
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_init_label_worker,
        initargs=(
            str(output_dir),
            cfg,
            ablation,
            require_interaction_heavy,
            collect_scene_metadata,
            skip_existing,
            compress,
            allow_scenario_ids,
            exclude_scenario_ids,
            profile_label_engine,
        ),
    ) as pool:
        while len(futures) < max_pending and submit_one(pool):
            pass
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"status": "error", "error": repr(exc), "traceback": traceback.format_exc(), "seconds": 0.0}
                    if fail_on_error:
                        _append_profile(profile_jsonl, res)
                        raise
                _advance_progress(iterator)
                handle_result(res, iterator)
            if limit is not None and count >= limit:
                stop_submit = True
                # Do not let a smoke-test limit overshoot by the whole pending queue.
                # Running tasks may still finish, but queued tasks are cancelled.
                for pending in list(futures):
                    if pending.cancel():
                        futures.remove(pending)
            while len(futures) < max_pending and not stop_submit and submit_one(pool):
                pass
    return count


def _format_rate(num: int, den: int) -> str:
    return f"{(100.0 * num / max(den, 1)):.3f}%"


def build_tfexample_id_index(
    tfexample_glob: str | list[str],
    output_jsonl: str | Path,
    *,
    progress: bool = True,
    max_examples_scanned: int | None = None,
) -> dict[str, object]:
    """Build a reusable scenario-id index for WOMD tf.Example shards.

    The merge stage often has very sparse labels.  Without an index it may scan
    tens or hundreds of thousands of tf.Example records before the first match.
    This index stores only lightweight metadata and lets later merges restrict
    scanning to the shards that contain target scenario ids.
    """
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    files = resolve_glob_patterns(tfexample_glob)
    scanned = 0
    bad_records = 0
    iterator = tqdm_iter(
        iter_tfexample_records_by_file(files),
        enabled=progress,
        total=max_examples_scanned,
        desc="Index WOMD tf.Example scenario ids",
        unit="example",
    )
    with output_jsonl.open("w", encoding="utf-8") as f:
        for filename, record_index, raw in iterator:
            scanned += 1
            try:
                parsed = parse_tfexample(raw)
                sid = scenario_id_from_parsed_tfexample(parsed)
                f.write(json.dumps({"scenario_id": sid, "file": filename, "record_index": int(record_index)}, ensure_ascii=False) + "\n")
            except Exception as exc:  # pragma: no cover - corrupt records are data dependent
                bad_records += 1
                f.write(json.dumps({"scenario_id": None, "file": filename, "record_index": int(record_index), "error": str(exc)}, ensure_ascii=False) + "\n")
            if hasattr(iterator, "set_postfix") and (scanned == 1 or scanned % 100 == 0):
                iterator.set_postfix(scanned=scanned, bad=bad_records, file=Path(filename).name[:24], refresh=True)
            if max_examples_scanned is not None and scanned >= max_examples_scanned:
                break
    return {"index": str(output_jsonl), "files": len(files), "scanned": scanned, "bad_records": bad_records}


def _records_for_label_ids_from_tfexample_index(index_jsonl: str | Path, label_ids: set[str]) -> tuple[dict[str, dict[int, str]], dict[str, object]]:
    """Map TFRecord files to exact record indices for requested label ids."""
    by_file: dict[str, dict[int, str]] = {}
    matched_ids: set[str] = set()
    sample_indexed_ids: list[str] = []
    sample_indexed_files: list[str] = []
    total_rows = 0
    index_jsonl = Path(index_jsonl)
    with index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            sid = row.get("scenario_id")
            if sid is not None and len(sample_indexed_ids) < 5:
                sample_indexed_ids.append(str(sid))
            fpath = row.get("file")
            if fpath is not None and len(sample_indexed_files) < 3 and str(fpath) not in sample_indexed_files:
                sample_indexed_files.append(str(fpath))
            if sid in label_ids and fpath is not None:
                try:
                    ridx = int(row.get("record_index", -1))
                except Exception:
                    ridx = -1
                if ridx >= 0:
                    matched_ids.add(str(sid))
                    by_file.setdefault(str(fpath), {})[ridx] = str(sid)
    stats: dict[str, object] = {
        "indexed_rows": total_rows,
        "indexed_label_matches": len(matched_ids),
        "indexed_files": len(by_file),
        "indexed_record_targets": sum(len(v) for v in by_file.values()),
        "sample_indexed_ids": sample_indexed_ids,
        "sample_indexed_files": sample_indexed_files,
    }
    return by_file, stats


def _files_for_label_ids_from_tfexample_index(index_jsonl: str | Path, label_ids: set[str]) -> tuple[list[str], dict[str, object]]:
    by_file, stats = _records_for_label_ids_from_tfexample_index(index_jsonl, label_ids)
    return sorted(by_file), stats


def _iter_indexed_target_tfexample_records(by_file: dict[str, dict[int, str]]):
    """Yield only target records using the existing {file, record_index} index.

    TFRecord files are still sequential, so this is not true random access, but
    it avoids parsing every non-target tf.train.Example just to discover its id.
    """
    for filename in sorted(by_file):
        targets = by_file[filename]
        if not targets:
            continue
        max_idx = max(targets)
        for fpath, record_index, raw in iter_tfexample_records_by_file([filename]):
            if int(record_index) in targets:
                yield targets[int(record_index)], raw
            if int(record_index) >= int(max_idx):
                break




def _waymax_missing_required_womd_keys(example: Mapping[str, object]) -> list[str]:
    """Return missing WOMD tf.Example keys that are required for cache-source Waymax replay.

    The replay code can synthesize ``state/all/*`` and ``traffic_light_state/all/*``
    from the split past/current/future tensors, so the cache only needs to retain
    the original split keys.  SDC route path samples are intentionally not hard
    requirements for safety labels: when they are absent, replay disables route
    paths and still computes collision/offroad labels.
    """
    required = [
        "scenario/id",
        "state/past/x",
        "state/current/x",
        "state/future/x",
        "state/past/y",
        "state/current/y",
        "state/future/y",
        "state/past/valid",
        "state/current/valid",
        "state/future/valid",
        "state/current/bbox_yaw",
        "state/current/length",
        "state/current/width",
        "state/current/height",
        "state/current/velocity_x",
        "state/current/velocity_y",
        "state/is_sdc",
        "roadgraph_samples/xyz",
    ]
    missing = [k for k in required if k not in example]
    # Some WOMD releases keep roadgraph coordinates split instead of xyz.
    if "roadgraph_samples/xyz" in missing and all(k in example for k in ("roadgraph_samples/x", "roadgraph_samples/y", "roadgraph_samples/z")):
        missing.remove("roadgraph_samples/xyz")
    return missing


def _merge_one_tfexample_to_cache(
    *,
    sid: str,
    parsed_example,
    label_path: str | Path,
    output_dir: str | Path,
    compress: bool,
    verify_cache: bool = False,
    require_waymax_ready: bool = False,
) -> dict[str, object]:
    """Decode one matched WOMD tf.Example and write the merged tensor cache item."""
    t0 = time.perf_counter()
    t_decode = time.perf_counter()
    example = decode_parsed_tfexample(parsed_example)
    decode_s = time.perf_counter() - t_decode

    waymax_missing = _waymax_missing_required_womd_keys(example)
    waymax_ready = len(waymax_missing) == 0
    if require_waymax_ready and not waymax_ready:
        raise KeyError(f"WOMD tf.Example for {sid} is missing required Waymax replay keys: {waymax_missing[:8]}")

    arrays: dict[str, object] = {}
    for key, val in example.items():
        safe = _npz_key(key)
        arrays[f"womd__{safe}"] = np.frombuffer(val, dtype=np.uint8) if isinstance(val, bytes) else np.asarray(val)
    arrays[_npz_key("cache/meta/waymax_ready")] = np.asarray(bool(waymax_ready))
    arrays[_npz_key("cache/meta/waymax_missing_keys")] = np.asarray(";".join(waymax_missing))

    t_label = time.perf_counter()
    restored_for_mask: dict[str, object] = {}
    with np.load(label_path, allow_pickle=True) as label:
        for key in label.files:
            arrays[_npz_key(key)] = label[key]
            restored_for_mask[key] = label[key]
    label_s = time.perf_counter() - t_label

    # Mask labels for critical agents that are not visible in WOMD tf.Example
    # padded state tensors.  This keeps future caches trainable even when the
    # Scenario proto contains > max_agents tracks and a selected critical index
    # exceeds the model input dimension.
    restored_for_mask.update({k.replace("__", "/"): v for k, v in arrays.items() if k.startswith("womd__") or k.startswith("state__")})
    t_mask = time.perf_counter()
    align_critical_agents_to_womd_input(restored_for_mask)
    mask_out_of_range_critical_agents(restored_for_mask)
    mask_s = time.perf_counter() - t_mask
    for key, val in restored_for_mask.items():
        if key.startswith("cowp/") or key.startswith("map/"):
            arrays[_npz_key(key)] = val

    out_path = Path(output_dir) / f"{sid}.npz"
    t_write = time.perf_counter()
    _write_npz(out_path, arrays, compress=compress)
    write_s = time.perf_counter() - t_write
    if verify_cache:
        with np.load(out_path, allow_pickle=True) as data:
            _ = data.files
    return {
        "status": "written",
        "scenario_id": sid,
        "output": str(out_path),
        "num_arrays": len(arrays),
        "decode_seconds": decode_s,
        "label_read_seconds": label_s,
        "mask_seconds": mask_s,
        "write_seconds": write_s,
        "seconds": time.perf_counter() - t0,
        "compress": bool(compress),
        "waymax_ready": bool(waymax_ready),
        "waymax_missing_keys": waymax_missing,
    }


def _init_tensor_cache_worker(
    label_paths: dict[str, str],
    output_dir: str,
    skip_existing: bool,
    compress: bool,
    verify_cache: bool,
    require_waymax_ready: bool,
) -> None:
    _CACHE_WORKER_STATE.clear()
    _CACHE_WORKER_STATE.update(
        {
            "label_paths": label_paths,
            "output_dir": output_dir,
            "skip_existing": bool(skip_existing),
            "compress": bool(compress),
            "verify_cache": bool(verify_cache),
            "require_waymax_ready": bool(require_waymax_ready),
        }
    )


def _build_tensor_cache_file_worker(filename: str) -> dict[str, object]:
    """Worker task: scan one tf.Example shard and write cache files for matching labels."""
    if not _CACHE_WORKER_STATE:
        raise RuntimeError("Tensor-cache worker state was not initialized.")
    label_paths: dict[str, str] = _CACHE_WORKER_STATE["label_paths"]  # type: ignore[assignment]
    output_dir = str(_CACHE_WORKER_STATE["output_dir"])
    skip_existing = bool(_CACHE_WORKER_STATE["skip_existing"])
    compress = bool(_CACHE_WORKER_STATE["compress"])
    verify_cache = bool(_CACHE_WORKER_STATE["verify_cache"])
    require_waymax_ready = bool(_CACHE_WORKER_STATE["require_waymax_ready"])
    scanned = 0
    matched = 0
    written = 0
    existing = 0
    errors = 0
    decode_s = 0.0
    write_s = 0.0
    waymax_not_ready = 0
    examples: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for _fpath, _record_index, raw in iter_tfexample_records_by_file([filename]):
        scanned += 1
        try:
            parsed = parse_tfexample(raw)
            sid = scenario_id_from_parsed_tfexample(parsed)
        except Exception as exc:
            errors += 1
            if len(examples) < 8:
                examples.append({"status": "error", "file": filename, "record_index": int(_record_index), "error": f"id_parse_failed: {exc}"})
            continue
        label_path = label_paths.get(str(sid))
        if label_path is None:
            continue
        matched += 1
        out_path = Path(output_dir) / f"{sid}.npz"
        if skip_existing and out_path.exists():
            existing += 1
            continue
        try:
            res = _merge_one_tfexample_to_cache(
                sid=str(sid),
                parsed_example=parsed,
                label_path=label_path,
                output_dir=output_dir,
                compress=compress,
                verify_cache=verify_cache,
                require_waymax_ready=require_waymax_ready,
            )
            written += 1
            decode_s += float(res.get("decode_seconds", 0.0))
            write_s += float(res.get("write_seconds", 0.0))
            if not bool(res.get("waymax_ready", False)):
                waymax_not_ready += 1
            if len(examples) < 8:
                examples.append({k: res[k] for k in ("status", "scenario_id", "output", "seconds", "waymax_ready") if k in res})
        except Exception as exc:
            errors += 1
            if len(examples) < 8:
                examples.append({"status": "error", "scenario_id": str(sid), "error": repr(exc)})
    return {
        "file": filename,
        "scanned": scanned,
        "matched": matched,
        "written": written,
        "existing": existing,
        "errors": errors,
        "decode_seconds": decode_s,
        "write_seconds": write_s,
        "waymax_not_ready": waymax_not_ready,
        "seconds": time.perf_counter() - t0,
        "examples": examples,
    }


def _build_tensor_cache_parallel_by_file(
    tfexample_glob: str | list[str],
    label_paths: dict[str, Path],
    output_dir: str | Path,
    *,
    progress: bool,
    verify_cache: bool,
    skip_existing: bool,
    compress: bool,
    profile_jsonl: str | Path | None,
    num_workers: int,
    start_method: str | None,
    require_waymax_ready: bool,
) -> int:
    """Parallel tensor-cache merge by TFRecord shard.

    This is the preferred from-scratch path when no reusable exact tf.Example
    index exists: every worker scans different TFRecord files, parses only the
    scenario id for non-target examples, and decodes/writes only matching labels.
    """
    files = resolve_glob_patterns(tfexample_glob)
    profile_path = Path(profile_jsonl) if profile_jsonl else None
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("", encoding="utf-8")
    workers = max(1, int(num_workers))
    mp_context = mp.get_context(start_method) if start_method else None
    label_paths_s = {str(k): str(v) for k, v in label_paths.items()}
    total_written = 0
    total_matched = 0
    total_scanned = 0
    total_existing = 0
    total_errors = 0
    waymax_not_ready = 0
    iterator = tqdm_iter(iter(int, 1), enabled=progress, total=len(files), desc=f"Parallel merge WOMD tf.Example cache ({workers} workers)", unit="file")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_init_tensor_cache_worker,
        initargs=(label_paths_s, str(output_dir), skip_existing, compress, verify_cache, require_waymax_ready),
    ) as pool:
        futures = [pool.submit(_build_tensor_cache_file_worker, f) for f in files]
        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            done_count += 1
            total_written += int(res.get("written", 0))
            total_matched += int(res.get("matched", 0))
            total_scanned += int(res.get("scanned", 0))
            total_existing += int(res.get("existing", 0))
            total_errors += int(res.get("errors", 0))
            waymax_not_ready += int(res.get("waymax_not_ready", 0))
            if profile_path is not None:
                with profile_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            _advance_progress(iterator)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    scanned=total_scanned,
                    matched=total_matched,
                    written=total_written,
                    existing=total_existing,
                    errors=total_errors,
                    waymax_not_ready=waymax_not_ready,
                    refresh=True,
                )
    if total_written == 0 and total_existing == 0:
        raise RuntimeError(
            "No tensor cache files were written in parallel merge. "
            f"Scanned {total_scanned} tf.Example records, matched {total_matched} labels. "
            "Check that --tfexample-glob and --labels-dir use the same WOMD split/version."
        )
    return total_written

def build_tensor_cache(
    tfexample_glob: str | list[str],
    labels_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    progress: bool = True,
    verify_cache: bool = False,
    skip_existing: bool = False,
    compress: bool = False,
    max_examples_scanned: int | None = None,
    profile_jsonl: str | Path | None = None,
    tfexample_index_jsonl: str | Path | None = None,
    build_index_if_missing: bool = False,
    num_workers: int = 1,
    start_method: str | None = None,
    require_waymax_ready: bool = False,
    prefer_parallel_scan: bool = False,
) -> int:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = {p.stem: p for p in sorted(labels_dir.glob("*.npz"))}
    if not label_paths:
        raise FileNotFoundError(f"No COWP label npz files found in {labels_dir}")
    if limit is not None:
        keep_ids = set(sorted(label_paths)[: int(limit)])
        label_paths = {sid: p for sid, p in label_paths.items() if sid in keep_ids}
    remaining = set(label_paths)
    target = len(label_paths)

    # From-scratch construction is usually faster as a parallel single-pass scan
    # than as "build full id index -> scan target records again".  Exact-index
    # mode remains useful when a complete reusable index already exists.
    if int(num_workers) > 1 and (prefer_parallel_scan or tfexample_index_jsonl is None):
        return _build_tensor_cache_parallel_by_file(
            tfexample_glob,
            label_paths,
            output_dir,
            progress=progress,
            verify_cache=verify_cache,
            skip_existing=skip_existing,
            compress=compress,
            profile_jsonl=profile_jsonl,
            num_workers=int(num_workers),
            start_method=start_method,
            require_waymax_ready=require_waymax_ready,
        )

    scan_glob: str | list[str] = tfexample_glob
    index_stats: dict[str, object] = {}
    indexed_record_targets: dict[str, dict[int, str]] | None = None
    if tfexample_index_jsonl is not None:
        index_path = Path(tfexample_index_jsonl)
        if build_index_if_missing or not index_path.exists():
            summary = build_tfexample_id_index(tfexample_glob, index_path, progress=progress)
            print(f"Built tf.Example id index: {json.dumps(summary, ensure_ascii=False)}")
        if index_path.exists():
            indexed_record_targets, index_stats = _records_for_label_ids_from_tfexample_index(index_path, set(label_paths))
            if indexed_record_targets:
                scan_glob = sorted(indexed_record_targets)
                print(
                    "Using exact tf.Example index targets: "
                    f"{index_stats['indexed_label_matches']} / {len(label_paths)} label ids found in "
                    f"{index_stats['indexed_files']} shard(s), "
                    f"{index_stats['indexed_record_targets']} exact record(s)."
                )
            else:
                preview = ", ".join(sorted(label_paths)[:5])
                indexed_preview = ", ".join(str(x) for x in index_stats.get("sample_indexed_ids", [])[:5])
                indexed_files_preview = ", ".join(Path(str(x)).name for x in index_stats.get("sample_indexed_files", [])[:3])
                raise RuntimeError(
                    "The tf.Example id index contains no rows matching the current label ids. "
                    f"First label ids: {preview}. First indexed tf.Example ids: {indexed_preview}. "
                    f"Indexed files preview: {indexed_files_preview}. "
                    "This usually means the labels and tf.Example glob use different WOMD split/version; "
                    "for validation labels, run 02_build_tensor_cache.py with --split validation or an explicit validation --tfexample-glob."
                )
    count = 0
    scanned = 0
    matched = 0
    skipped_existing = 0
    decode_seconds = 0.0
    write_seconds = 0.0
    written_paths: list[Path] = []
    if profile_jsonl:
        profile_path = Path(profile_jsonl)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("", encoding="utf-8")
    else:
        profile_path = None

    if indexed_record_targets:
        record_iter = _iter_indexed_target_tfexample_records(indexed_record_targets)
        total_iter = min(len(label_paths), int(index_stats.get("indexed_record_targets", 0) or 0))
        iterator = tqdm_iter(
            record_iter,
            enabled=progress,
            total=total_iter,
            desc="Merge exact indexed WOMD tf.Example records",
            unit="example",
        )
        exact_index_mode = True
    else:
        iterator = tqdm_iter(
            iter_tfexample_records(scan_glob),
            enabled=progress,
            total=max_examples_scanned,
            desc="Scan WOMD tf.Example records and merge matching labels",
            unit="example",
        )
        exact_index_mode = False

    def update_postfix(stage: str, sid: str = "") -> None:
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                stage=stage,
                scanned=scanned,
                matched=matched,
                hit=_format_rate(matched, scanned),
                written=count,
                remaining=len(remaining),
                existing=skipped_existing,
                decode_s=f"{decode_seconds:.1f}",
                write_s=f"{write_seconds:.1f}",
                last=sid[:10],
                refresh=True,
            )

    for item in iterator:
        scanned += 1
        if exact_index_mode:
            sid, raw = item
            sid = str(sid)
            parsed = parse_tfexample(raw)
        else:
            raw = item
            parsed = parse_tfexample(raw)
            sid = scenario_id_from_parsed_tfexample(parsed)
        label_path = label_paths.get(sid)
        if label_path is None:
            if scanned == 1 or scanned % 100 == 0:
                update_postfix("seeking_match", sid)
            if max_examples_scanned is not None and scanned >= max_examples_scanned:
                break
            continue
        matched += 1
        out_path = output_dir / f"{sid}.npz"
        if skip_existing and out_path.exists():
            skipped_existing += 1
            remaining.discard(sid)
            update_postfix("skip_existing", sid)
            if len(remaining) == 0 or (limit is not None and count >= target):
                break
            if max_examples_scanned is not None and scanned >= max_examples_scanned:
                break
            continue

        res = _merge_one_tfexample_to_cache(
            sid=sid,
            parsed_example=parsed,
            label_path=label_path,
            output_dir=output_dir,
            compress=compress,
            verify_cache=False,
            require_waymax_ready=require_waymax_ready,
        )
        decode_seconds += float(res.get("decode_seconds", 0.0))
        write_seconds += float(res.get("write_seconds", 0.0))
        if profile_path is not None:
            row = dict(res)
            row["scanned"] = scanned
            row["written_index"] = count + 1
            with profile_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written_paths.append(out_path)
        remaining.discard(sid)
        count += 1
        update_postfix("wrote", sid)
        if limit is not None and count >= target:
            break
        if len(remaining) == 0:
            break
        if max_examples_scanned is not None and scanned >= max_examples_scanned:
            break
    if verify_cache:
        for p in tqdm_iter(written_paths, enabled=progress, desc="Verify written tensor cache", unit="file"):
            with np.load(p, allow_pickle=True) as data:
                _ = data.files
    if count == 0:
        missing_preview = ", ".join(sorted(remaining)[:5])
        index_hint = ""
        if tfexample_index_jsonl is not None:
            index_hint = f" tf.Example index stats: {index_stats}."
        raise RuntimeError(
            "No merged tensor cache files were written. "
            f"Scanned {scanned} tf.Example records and found {matched} label id matches. "
            f"First missing label ids: {missing_preview}. Check that --tfexample-glob and --labels-dir use the same WOMD split/version/shard subset."
            + index_hint
        )
    return count
