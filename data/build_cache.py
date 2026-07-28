from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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


def _files_for_label_ids_from_tfexample_index(index_jsonl: str | Path, label_ids: set[str]) -> tuple[list[str], dict[str, object]]:
    files: set[str] = set()
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
            if sid in label_ids:
                matched_ids.add(str(sid))
                files.add(str(row.get("file")))
    stats: dict[str, object] = {
        "indexed_rows": total_rows,
        "indexed_label_matches": len(matched_ids),
        "indexed_files": len(files),
        "sample_indexed_ids": sample_indexed_ids,
        "sample_indexed_files": sample_indexed_files,
    }
    return sorted(files), stats


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
) -> int:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = {p.stem: p for p in sorted(labels_dir.glob("*.npz"))}
    if not label_paths:
        raise FileNotFoundError(f"No COWP label npz files found in {labels_dir}")
    remaining = set(label_paths)
    target = min(int(limit), len(label_paths)) if limit is not None else len(label_paths)

    scan_glob: str | list[str] = tfexample_glob
    index_stats: dict[str, int] = {}
    if tfexample_index_jsonl is not None:
        index_path = Path(tfexample_index_jsonl)
        if build_index_if_missing or not index_path.exists():
            summary = build_tfexample_id_index(tfexample_glob, index_path, progress=progress)
            print(f"Built tf.Example id index: {json.dumps(summary, ensure_ascii=False)}")
        if index_path.exists():
            indexed_files, index_stats = _files_for_label_ids_from_tfexample_index(index_path, set(label_paths))
            if indexed_files:
                scan_glob = indexed_files
                print(
                    "Using tf.Example index: "
                    f"{index_stats['indexed_label_matches']} / {len(label_paths)} label ids found in "
                    f"{index_stats['indexed_files']} shard(s)."
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

    iterator = tqdm_iter(
        iter_tfexample_records(scan_glob),
        enabled=progress,
        total=max_examples_scanned,
        desc="Scan WOMD tf.Example records and merge matching labels",
        unit="example",
    )

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

    for raw in iterator:
        scanned += 1
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

        t_decode = time.perf_counter()
        example = decode_parsed_tfexample(parsed)
        decode_seconds += time.perf_counter() - t_decode
        arrays = {}
        for key, val in example.items():
            safe = _npz_key(key)
            arrays[f"womd__{safe}"] = np.frombuffer(val, dtype=np.uint8) if isinstance(val, bytes) else np.asarray(val)
        with np.load(label_path, allow_pickle=True) as label:
            restored_for_mask = {}
            for key in label.files:
                arrays[_npz_key(key)] = label[key]
                restored_for_mask[key] = label[key]
        # Mask labels for critical agents that are not visible in WOMD tf.Example
        # padded state tensors.  This keeps future caches trainable even when the
        # Scenario proto contains > max_agents tracks and a selected critical index
        # exceeds the model input dimension.
        restored_for_mask.update({k.replace("__", "/"): v for k, v in arrays.items() if k.startswith("womd__") or k.startswith("state__")})
        align_critical_agents_to_womd_input(restored_for_mask)
        mask_out_of_range_critical_agents(restored_for_mask)
        for key, val in restored_for_mask.items():
            if key.startswith("cowp/") or key.startswith("map/"):
                arrays[_npz_key(key)] = val

        t_write = time.perf_counter()
        _write_npz(out_path, arrays, compress=compress)
        dt_write = time.perf_counter() - t_write
        write_seconds += dt_write
        if profile_path is not None:
            with profile_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "scenario_id": sid,
                    "output": str(out_path),
                    "scanned": scanned,
                    "written_index": count + 1,
                    "num_arrays": len(arrays),
                    "write_seconds": dt_write,
                    "compress": bool(compress),
                }, ensure_ascii=False) + "\n")
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
