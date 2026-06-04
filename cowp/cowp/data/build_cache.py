from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from cowp.data.parse_scenario_proto import iter_scenarios, scenario_to_scene
from cowp.data.parse_tfexample import iter_tfexamples, scenario_id_from_tfexample
from cowp.label.label_engine import build_labels_for_scene
from cowp.label.scene_filter import is_interaction_heavy


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


def build_labels_from_proto(proto_glob: str | list[str], output_dir: str | Path, cfg: dict, limit: int | None = None, ablation: dict | None = None) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    filter_cfg = cfg.get("dataset", {})
    require_interaction_heavy = bool(filter_cfg.get("require_interaction_heavy", False))
    for scenario in iter_scenarios(proto_glob):
        scene = scenario_to_scene(scenario, keep_raw=False)
        if require_interaction_heavy:
            heavy, _meta = is_interaction_heavy(scene, cfg)
            if not heavy:
                continue
        label = build_labels_for_scene(scene, cfg, ablation=ablation)
        save_label_npz(label, output_dir / f"{scene.scenario_id}.npz")
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def build_tensor_cache(tfexample_glob: str | list[str], labels_dir: str | Path, output_dir: str | Path, limit: int | None = None) -> int:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for example in iter_tfexamples(tfexample_glob):
        sid = scenario_id_from_tfexample(example)
        label_path = labels_dir / f"{sid}.npz"
        if not label_path.exists():
            continue
        label = np.load(label_path, allow_pickle=True)
        arrays = {}
        for key, val in example.items():
            safe = _npz_key(key)
            arrays[f"womd__{safe}"] = np.frombuffer(val, dtype=np.uint8) if isinstance(val, bytes) else np.asarray(val)
        for key in label.files:
            arrays[_npz_key(key)] = label[key]
        np.savez_compressed(output_dir / f"{sid}.npz", **arrays)
        count += 1
        if limit is not None and count >= limit:
            break
    # Finalization check: load every file once to catch corrupt caches.
    for p in output_dir.glob("*.npz"):
        with np.load(p, allow_pickle=True) as data:
            _ = data.files
    return count
