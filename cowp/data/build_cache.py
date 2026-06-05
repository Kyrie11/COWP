from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from cowp.data.parse_scenario_proto import iter_scenarios, scenario_to_scene
from cowp.data.parse_tfexample import iter_tfexamples, scenario_id_from_tfexample
from cowp.label.label_engine import build_labels_for_scene
from cowp.label.scene_filter import is_interaction_heavy
from utils.progress import tqdm_iter


def _npz_key(key: str) -> str:
    return key.replace("/", "__")


def save_label_npz(label: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in label.items():
        if isinstance(v, str):
            arrays[k] = np.asarray(v)
        else:
            arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def build_labels_from_proto(
    proto_glob: str | list[str],
    output_dir: str | Path,
    cfg: dict,
    limit: int | None = None,
    ablation: dict | None = None,
    progress: bool = True,
    skip_existing: bool = False,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_filter = 0
    skipped_existing = 0
    filter_cfg = cfg.get("dataset", {})
    require_interaction_heavy = bool(filter_cfg.get("require_interaction_heavy", False))
    collect_scene_metadata = bool(filter_cfg.get("collect_scene_metadata", True))
    iterator = tqdm_iter(
        iter_scenarios(proto_glob),
        enabled=progress,
        desc="Build COWP labels from Scenario protos",
        unit="scenario",
    )
    for scenario in iterator:
        scene = scenario_to_scene(scenario, keep_raw=False)
        heavy_meta: dict[str, object] | None = None
        if require_interaction_heavy or collect_scene_metadata:
            heavy, meta = is_interaction_heavy(scene, cfg)
            heavy_meta = dict(meta)
            heavy_meta["interaction_heavy"] = bool(heavy)
            if require_interaction_heavy and not heavy:
                skipped_filter += 1
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(written=count, filtered=skipped_filter, existing=skipped_existing)
                continue
        label_path = output_dir / f"{scene.scenario_id}.npz"
        if skip_existing and label_path.exists():
            skipped_existing += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(written=count, filtered=skipped_filter, existing=skipped_existing)
            continue
        label = build_labels_for_scene(scene, cfg, ablation=ablation, scene_meta=heavy_meta)
        save_label_npz(label, label_path)
        count += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(written=count, filtered=skipped_filter, existing=skipped_existing)
        if limit is not None and count >= limit:
            break
    return count


def build_tensor_cache(
    tfexample_glob: str | list[str],
    labels_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    progress: bool = True,
    verify_cache: bool = False,
    skip_existing: bool = False,
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
        np.savez_compressed(out_path, **arrays)
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
