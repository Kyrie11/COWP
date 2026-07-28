from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from cowp.utils.progress import tqdm_iter


def stress_manifest_from_label(label: Mapping[str, np.ndarray], scenario_id: str) -> dict[str, object]:
    cand_valid = np.asarray(label.get("cowp/candidates/valid", np.ones_like(label["cowp/candidates/false_safe"])), dtype=bool)
    false_safe = np.asarray(label["cowp/candidates/false_safe"], dtype=bool) & cand_valid
    ncf = np.asarray(label["cowp/candidates/noncoercive_feasible"], dtype=bool) & cand_valid
    conventional = np.asarray(label.get("cowp/candidates/conventional_safe", np.zeros_like(false_safe)), dtype=bool) & cand_valid
    witness = np.asarray(label.get("cowp/witness/exists", np.zeros((len(false_safe), 0))), dtype=bool)
    return {
        "root_scene_id": scenario_id,
        "noncoercive_candidate_indices": np.where(ncf)[0].astype(int).tolist(),
        "coercive_candidate_indices": np.where(false_safe)[0].astype(int).tolist(),
        "conventional_safe_candidate_indices": np.where(conventional)[0].astype(int).tolist(),
        "num_noncoercive_candidates": int(np.sum(ncf)),
        "num_false_safe_candidates": int(np.sum(false_safe)),
        "num_conventional_safe_candidates": int(np.sum(conventional)),
        "num_positive_witness_pairs": int(np.sum(witness)),
        "expected_accept_noncoercive": bool(np.any(ncf)),
        "expected_reject_false_safe": bool(np.any(false_safe)),
    }


def write_stress_manifest(labels_dir: str | Path, output_jsonl: str | Path, progress: bool = True) -> int:
    labels_dir = Path(labels_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(labels_dir.glob("*.npz"))
    written = 0
    scanned = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc="Build false-safe stress manifest", unit="file")
        for path in iterator:
            scanned += 1
            with np.load(path, allow_pickle=True) as data:
                sid = str(data["scenario/id"].item()) if "scenario/id" in data else path.stem
                # Load only the small manifest fields; the trajectory/response tensors
                # dominate NPZ size and are unnecessary for this pass.
                label = {
                    "cowp/candidates/false_safe": data["cowp/candidates/false_safe"],
                    "cowp/candidates/noncoercive_feasible": data["cowp/candidates/noncoercive_feasible"],
                    "cowp/candidates/valid": data["cowp/candidates/valid"] if "cowp/candidates/valid" in data else np.ones_like(data["cowp/candidates/false_safe"], dtype=bool),
                    "cowp/candidates/conventional_safe": data["cowp/candidates/conventional_safe"] if "cowp/candidates/conventional_safe" in data else np.zeros_like(data["cowp/candidates/false_safe"], dtype=bool),
                    "cowp/witness/exists": data["cowp/witness/exists"] if "cowp/witness/exists" in data else np.zeros((len(data["cowp/candidates/false_safe"]), 0), dtype=bool),
                }
                item = stress_manifest_from_label(label, sid)
            if item["expected_accept_noncoercive"] and item["expected_reject_false_safe"]:
                f.write(json.dumps(item) + "\n")
                written += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(scanned=scanned, written=written)
    summary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump({"scanned_label_files": scanned, "stress_manifest_rows": written}, sf, indent=2)
    return written
