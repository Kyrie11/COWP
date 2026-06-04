from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def stress_manifest_from_label(label: dict[str, np.ndarray], scenario_id: str) -> dict[str, object]:
    false_safe = np.asarray(label["cowp/candidates/false_safe"], dtype=bool)
    ncf = np.asarray(label["cowp/candidates/noncoercive_feasible"], dtype=bool)
    return {
        "root_scene_id": scenario_id,
        "noncoercive_candidate_indices": np.where(ncf)[0].astype(int).tolist(),
        "coercive_candidate_indices": np.where(false_safe)[0].astype(int).tolist(),
        "expected_accept_noncoercive": bool(np.any(ncf)),
        "expected_reject_false_safe": bool(np.any(false_safe)),
    }


def write_stress_manifest(labels_dir: str | Path, output_jsonl: str | Path) -> None:
    labels_dir = Path(labels_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for path in sorted(labels_dir.glob("*.npz")):
            data = np.load(path, allow_pickle=True)
            sid = str(data["scenario/id"].item()) if "scenario/id" in data else path.stem
            item = stress_manifest_from_label({k: data[k] for k in data.files}, sid)
            if item["expected_accept_noncoercive"] and item["expected_reject_false_safe"]:
                f.write(json.dumps(item) + "\n")
