from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cowp.data.parse_scenario_proto import _import_scenario_proto, _import_tensorflow, resolve_glob_patterns as resolve_scenario_files
from cowp.data.parse_tfexample import resolve_glob_patterns as resolve_tfexample_files
from importlib import import_module

_shard_manifest = import_module("cowp.scripts.64_validate_womd_v131_contract")._shard_manifest

STANDARD_EXPECTED = {"training": 1000, "validation": 150, "testing": 150}
KNOWN_SPLITS = (
    "training",
    "validation",
    "testing",
    "validation_interactive",
    "testing_interactive",
    "training_20s",
)


def _split_role(split: str) -> dict[str, object]:
    if split == "training":
        return {
            "role": "primary_train",
            "use_for_cowp": True,
            "future_labels_expected": True,
            "independent_benchmark": False,
            "note": "Primary COWP training source. Scenario proto is authoritative for labels; tf.Example is the matched model/Waymax tensor source.",
        }
    if split == "validation":
        return {
            "role": "primary_validation",
            "use_for_cowp": True,
            "future_labels_expected": True,
            "independent_benchmark": True,
            "note": "Primary held-out COWP validation/probe source. Keep disjoint from training by scenario_id.",
        }
    if split == "testing":
        return {
            "role": "official_blind_test",
            "use_for_cowp": False,
            "future_labels_expected": False,
            "independent_benchmark": True,
            "note": "WOMD test future ground truth is hidden; do not build natural/transport/witness labels or log-playback Waymax counterfactual supervision from it.",
        }
    if split == "validation_interactive":
        return {
            "role": "interaction_challenge_stress_validation",
            "use_for_cowp": "secondary_only",
            "future_labels_expected": True,
            "independent_benchmark": False,
            "note": "Challenge-curated interaction stress split. Use as a secondary stress/stratified benchmark, not as a replacement for standard validation and not as independent evidence until scenario-id overlap with validation is audited.",
        }
    if split == "testing_interactive":
        return {
            "role": "interaction_challenge_blind_test",
            "use_for_cowp": False,
            "future_labels_expected": False,
            "independent_benchmark": False,
            "note": "Challenge-specific blind split; no offline COWP mechanism labels from hidden future.",
        }
    return {
        "role": "auxiliary_release_split",
        "use_for_cowp": False,
        "future_labels_expected": None,
        "independent_benchmark": False,
        "note": "Auxiliary WOMD release directory; not part of the primary COWP train/validation benchmark contract unless explicitly justified.",
    }


def _scenario_sample_stats(files: list[str], sample_shards: int) -> dict[str, object]:
    if not files:
        return {"sampled_records": 0}
    scenario_pb2 = _import_scenario_proto()
    tf = _import_tensorflow()
    n = min(max(int(sample_shards), 1), len(files))
    idx = sorted(set(np.linspace(0, len(files) - 1, num=n, dtype=np.int64).tolist()))
    selected = [files[i] for i in idx]
    rows: list[dict[str, object]] = []
    for filename in selected:
        raw = next(iter(tf.data.TFRecordDataset([filename]).take(1)), None)
        if raw is None:
            continue
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(bytes(raw.numpy()))
        current = int(sc.current_time_index)
        future_steps = max(0, len(sc.timestamps_seconds) - current - 1)
        sdc_future_valid = 0
        if 0 <= int(sc.sdc_track_index) < len(sc.tracks):
            sdc = sc.tracks[int(sc.sdc_track_index)]
            sdc_future_valid = sum(bool(x.valid) for x in sdc.states[current + 1 :])
        rows.append(
            {
                "scenario_id": str(sc.scenario_id),
                "timestamps": len(sc.timestamps_seconds),
                "current_time_index": current,
                "future_steps": future_steps,
                "objects_of_interest": len(sc.objects_of_interest),
                "tracks_to_predict": len(sc.tracks_to_predict),
                "sdc_future_valid_steps": sdc_future_valid,
            }
        )
    if not rows:
        return {"sampled_records": 0}
    return {
        "sampled_records": len(rows),
        "timestamps_values": sorted({int(x["timestamps"]) for x in rows}),
        "current_time_index_values": sorted({int(x["current_time_index"]) for x in rows}),
        "future_steps_values": sorted({int(x["future_steps"]) for x in rows}),
        "objects_of_interest_mean": float(np.mean([int(x["objects_of_interest"]) for x in rows])),
        "objects_of_interest_positive_rate": float(np.mean([int(x["objects_of_interest"]) > 0 for x in rows])),
        "objects_of_interest_max": max(int(x["objects_of_interest"]) for x in rows),
        "tracks_to_predict_mean": float(np.mean([int(x["tracks_to_predict"]) for x in rows])),
        "tracks_to_predict_max": max(int(x["tracks_to_predict"]) for x in rows),
        "sdc_future_valid_mean": float(np.mean([int(x["sdc_future_valid_steps"]) for x in rows])),
        "scenario_id_examples": [str(x["scenario_id"]) for x in rows[:10]],
    }


def _all_scenario_ids(files: Iterable[str]) -> set[str]:
    scenario_pb2 = _import_scenario_proto()
    tf = _import_tensorflow()
    ids: set[str] = set()
    for filename in files:
        for raw in tf.data.TFRecordDataset([filename]):
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(bytes(raw.numpy()))
            ids.add(str(sc.scenario_id))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit local WOMD 1.3.1 split layout and challenge-specialized split relationships before COWP dataset construction.")
    ap.add_argument("--womd-root", required=True)
    ap.add_argument("--sample-scenario-shards", type=int, default=16)
    ap.add_argument("--full-validation-interactive-overlap", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.womd_root)
    result: dict[str, object] = {
        "schema_version": "cowp_womd_v1_3_1_split_layout_v1",
        "womd_root": str(root.resolve()),
        "splits": {},
        "benchmark_recommendation": {
            "primary_train": "scenario/training + matched tf_example/training",
            "primary_validation": "scenario/validation + matched tf_example/validation",
            "secondary_interaction_stress": "validation_interactive only after overlap audit; report separately from standard validation",
            "do_not_use_for_offline_mechanism_labels": ["testing", "testing_interactive"],
        },
    }

    split_rows: dict[str, object] = {}
    for split in KNOWN_SPLITS:
        scenario_glob = str(root / "uncompressed" / "scenario" / split / "*.tfrecord*")
        tf_glob = str(root / "uncompressed" / "tf_example" / split / "*.tfrecord*")
        scenario_files = resolve_scenario_files(scenario_glob)
        tf_files = resolve_tfexample_files(tf_glob)
        expected = STANDARD_EXPECTED.get(split)
        row: dict[str, object] = {
            **_split_role(split),
            "scenario": {
                "glob": scenario_glob,
                "shards": _shard_manifest(scenario_files, expected=expected),
                "sample": _scenario_sample_stats(scenario_files, args.sample_scenario_shards) if scenario_files else {"sampled_records": 0},
            },
            "tf_example": {
                "glob": tf_glob,
                "shards": _shard_manifest(tf_files, expected=expected),
            },
        }
        split_rows[split] = row
    result["splits"] = split_rows

    if args.full_validation_interactive_overlap:
        val_files = resolve_scenario_files(str(root / "uncompressed" / "scenario" / "validation" / "*.tfrecord*"))
        int_files = resolve_scenario_files(str(root / "uncompressed" / "scenario" / "validation_interactive" / "*.tfrecord*"))
        if val_files and int_files:
            val_ids = _all_scenario_ids(val_files)
            int_ids = _all_scenario_ids(int_files)
            overlap = val_ids & int_ids
            result["validation_interactive_overlap"] = {
                "validation_scenarios": len(val_ids),
                "validation_interactive_scenarios": len(int_ids),
                "overlap": len(overlap),
                "interactive_subset_of_validation": bool(int_ids <= val_ids),
                "validation_subset_of_interactive": bool(val_ids <= int_ids),
                "jaccard": float(len(overlap) / max(len(val_ids | int_ids), 1)),
                "overlap_examples": sorted(overlap)[:20],
            }
        else:
            result["validation_interactive_overlap"] = {"available": False}

    # Primary local completeness is a hard requirement for COWP. Auxiliary
    # interactive/test folders are reported but do not make this audit fail.
    primary_checks: dict[str, bool] = {}
    for split in ("training", "validation"):
        row = split_rows[split]
        primary_checks[f"scenario_{split}_complete"] = bool(row["scenario"]["shards"]["complete"])
        primary_checks[f"tf_example_{split}_complete"] = bool(row["tf_example"]["shards"]["complete"])
    result["primary_checks"] = primary_checks
    result["pass_primary_layout"] = all(primary_checks.values())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass_primary_layout"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
