from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Mapping

import numpy as np

from cowp.data.parse_scenario_proto import iter_scenario_records, iter_scenarios, scenario_to_scene
from cowp.data.parse_tfexample import iter_tfexamples, scenario_id_from_tfexample
from cowp.geometry.lane_graph import build_conflict_regions
from cowp.label.label_engine import build_labels_for_scene
from cowp.label.scene_filter import is_interaction_heavy
from cowp.utils.progress import tqdm_iter


def _npz_key(key: str) -> str:
    return key.replace("/", "__")


def _write_npz(path: str | Path, arrays: Mapping[str, object], *, compress: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


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
) -> dict[str, object]:
    t0 = time.perf_counter()
    scenario_pb2 = _import_scenario_proto()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw)
    sid = str(scenario.scenario_id)
    label_path = Path(output_dir) / f"{sid}.npz"
    if skip_existing and label_path.exists():
        return {"status": "existing", "scenario_id": sid, "seconds": time.perf_counter() - t0}

    scene = scenario_to_scene(scenario, keep_raw=False)
    regions = None
    heavy_meta: dict[str, object] | None = None
    if require_interaction_heavy or collect_scene_metadata:
        regions = build_conflict_regions(scene.map_data, cfg)
        heavy, meta = is_interaction_heavy(scene, cfg, conflict_regions=regions)
        heavy_meta = dict(meta)
        heavy_meta["interaction_heavy"] = bool(heavy)
        if require_interaction_heavy and not heavy:
            return {"status": "filtered", "scenario_id": sid, "seconds": time.perf_counter() - t0}
    label = build_labels_for_scene(scene, cfg, ablation=ablation, scene_meta=heavy_meta, conflict_regions=regions)
    save_label_npz(label, label_path, compress=compress)
    return {"status": "written", "scenario_id": sid, "path": str(label_path), "seconds": time.perf_counter() - t0}


def _append_profile(profile_jsonl: str | Path | None, result: Mapping[str, object]) -> None:
    if not profile_jsonl:
        return
    p = Path(profile_jsonl)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(result), ensure_ascii=False) + "\n")


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
    filter_cfg = cfg.get("dataset", {})
    require_interaction_heavy = bool(filter_cfg.get("require_interaction_heavy", False))
    collect_scene_metadata = bool(filter_cfg.get("collect_scene_metadata", True))
    total = max_scenarios_scanned
    desc = "Build COWP labels from Scenario protos"

    def handle_result(res: Mapping[str, object], iterator) -> None:
        nonlocal count, skipped_filter, skipped_existing
        status = str(res.get("status", "unknown"))
        if status == "written":
            count += 1
        elif status == "filtered":
            skipped_filter += 1
        elif status == "existing":
            skipped_existing += 1
        _append_profile(profile_jsonl, res)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                scanned=scanned,
                written=count,
                filtered=skipped_filter,
                existing=skipped_existing,
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
            )
            handle_result(res, iterator)
            if limit is not None and count >= limit:
                break
            if max_scenarios_scanned is not None and scanned >= max_scenarios_scanned:
                break
        return count

    workers = max(1, int(num_workers))
    max_pending = max(workers * 2, workers)
    raw_iter = iter(iter_scenario_records(proto_glob))
    futures = set()
    stop_submit = False
    iterator = tqdm_iter(range(max_scenarios_scanned or 10**12), enabled=progress, total=total, desc=f"{desc} ({workers} workers)", unit="scenario")

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
            _build_one_label_from_raw,
            raw,
            str(output_dir),
            cfg,
            ablation,
            require_interaction_heavy,
            collect_scene_metadata,
            skip_existing,
            compress,
        )
        futures.add(fut)
        return True

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(futures) < max_pending and submit_one(pool):
            pass
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                try:
                    next(iterator)
                except Exception:
                    pass
                handle_result(res, iterator)
            if limit is not None and count >= limit:
                stop_submit = True
            while len(futures) < max_pending and not stop_submit and submit_one(pool):
                pass
    return count


def build_tensor_cache(
    tfexample_glob: str | list[str],
    labels_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    progress: bool = True,
    verify_cache: bool = False,
    skip_existing: bool = False,
    compress: bool = True,
) -> int:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = {p.stem: p for p in sorted(labels_dir.glob("*.npz"))}
    remaining = set(label_paths)
    target = min(int(limit), len(label_paths)) if limit is not None else len(label_paths)
    count = 0
    scanned = 0
    skipped_existing = 0
    written_paths: list[Path] = []
    iterator = tqdm_iter(
        iter_tfexamples(tfexample_glob),
        enabled=progress,
        desc="Merge WOMD tf.Example with COWP labels",
        unit="example",
    )
    for example in iterator:
        scanned += 1
        sid = scenario_id_from_tfexample(example)
        label_path = label_paths.get(sid)
        if label_path is None:
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(scanned=scanned, written=count, remaining=len(remaining), existing=skipped_existing)
            continue
        out_path = output_dir / f"{sid}.npz"
        if skip_existing and out_path.exists():
            skipped_existing += 1
            remaining.discard(sid)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(scanned=scanned, written=count, remaining=len(remaining), existing=skipped_existing)
            if len(remaining) == 0 or (limit is not None and count >= target):
                break
            continue
        arrays = {}
        for key, val in example.items():
            safe = _npz_key(key)
            arrays[f"womd__{safe}"] = np.frombuffer(val, dtype=np.uint8) if isinstance(val, bytes) else np.asarray(val)
        with np.load(label_path, allow_pickle=True) as label:
            for key in label.files:
                arrays[_npz_key(key)] = label[key]
        _write_npz(out_path, arrays, compress=compress)
        written_paths.append(out_path)
        remaining.discard(sid)
        count += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(scanned=scanned, written=count, remaining=len(remaining), existing=skipped_existing)
        if limit is not None and count >= target:
            break
        if len(remaining) == 0:
            break
    if verify_cache:
        for p in tqdm_iter(written_paths, enabled=progress, desc="Verify written tensor cache", unit="file"):
            with np.load(p, allow_pickle=True) as data:
                _ = data.files
    return count
