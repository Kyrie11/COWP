from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from cowp.data.parse_scenario_proto import (
    _import_scenario_proto,
    _import_tensorflow,
    resolve_glob_patterns as resolve_scenario_files,
)
from cowp.data.parse_tfexample import resolve_glob_patterns as resolve_tfexample_files
from importlib import import_module

_shard_manifest = import_module("cowp.scripts.64_validate_womd_v131_contract")._shard_manifest

# WOMD v1.3.1 release layout is representation-specific.  Do NOT form a
# Cartesian product between split names and representations: in particular,
# scenario/training_20s and scenario/visualization have no tf.Example peers in
# the public layout used by COWP.
#
# Primary 9-second model-development contract (Waymo Motion docs / Waymax):
#   scenario/{training,validation} + tf_example/{training,validation}
# Testing is inventory/blind-evaluation only because future ground truth is
# hidden. Interactive splits are optional challenge/stress data and must be
# reported separately from the primary validation benchmark.
SCENARIO_SPLITS: dict[str, dict[str, object]] = {
    "training": {"expected_shards": 1000, "required_primary": True},
    "validation": {"expected_shards": 150, "required_primary": True},
    "testing": {"expected_shards": 150, "required_primary": False},
    "validation_interactive": {"expected_shards": None, "required_primary": False},
    "testing_interactive": {"expected_shards": None, "required_primary": False},
    "training_20s": {"expected_shards": None, "required_primary": False},
    "visualization": {"expected_shards": None, "required_primary": False},
}

TFEXAMPLE_SPLITS: dict[str, dict[str, object]] = {
    "training": {"expected_shards": 1000, "required_primary": True},
    "validation": {"expected_shards": 150, "required_primary": True},
    "testing": {"expected_shards": 150, "required_primary": False},
    "validation_interactive": {"expected_shards": None, "required_primary": False},
    "testing_interactive": {"expected_shards": None, "required_primary": False},
}

ALL_SPLITS = tuple(
    dict.fromkeys([*SCENARIO_SPLITS.keys(), *TFEXAMPLE_SPLITS.keys()])
)


def _split_role(split: str) -> dict[str, object]:
    if split == "training":
        return {
            "role": "primary_train",
            "use_for_cowp": True,
            "future_labels_expected": True,
            "independent_benchmark": False,
            "note": (
                "Primary COWP training source. Use scenario/training as the "
                "authoritative label/map/traffic-control source and matched "
                "tf_example/training for model tensors and Waymax."
            ),
        }
    if split == "validation":
        return {
            "role": "primary_validation",
            "use_for_cowp": True,
            "future_labels_expected": True,
            "independent_benchmark": True,
            "note": (
                "Primary held-out COWP validation/smoke/strict source. Use "
                "scenario/validation plus matched tf_example/validation."
            ),
        }
    if split == "testing":
        return {
            "role": "official_blind_test",
            "use_for_cowp": "blind_physical_evaluation_only",
            "future_labels_expected": False,
            "independent_benchmark": True,
            "note": (
                "Official blind test. Future ground truth is hidden, so do not "
                "construct natural/transport/witness/NCF labels from testing."
            ),
        }
    if split == "validation_interactive":
        return {
            "role": "interaction_challenge_stress_validation",
            "use_for_cowp": "secondary_only",
            "future_labels_expected": True,
            "independent_benchmark": False,
            "note": (
                "Optional interaction-focused validation/stress split. Keep it "
                "separate from standard validation and audit scenario-id overlap "
                "before making independence claims."
            ),
        }
    if split == "testing_interactive":
        return {
            "role": "interaction_challenge_blind_test",
            "use_for_cowp": "blind_interaction_evaluation_only",
            "future_labels_expected": False,
            "independent_benchmark": False,
            "note": (
                "Optional interaction challenge blind split. Do not construct "
                "future-dependent COWP mechanism labels from it."
            ),
        }
    if split == "training_20s":
        return {
            "role": "scenario_only_20s_auxiliary_source",
            "use_for_cowp": False,
            "future_labels_expected": None,
            "independent_benchmark": False,
            "note": (
                "Scenario-only auxiliary 20-second source. There is no "
                "tf_example/training_20s peer in the release layout used here. "
                "The current COWP 91-step pipeline must not mix it into primary "
                "training without a separate windowing/group-split design."
            ),
        }
    if split == "visualization":
        return {
            "role": "scenario_only_visualization_auxiliary",
            "use_for_cowp": False,
            "future_labels_expected": None,
            "independent_benchmark": False,
            "note": (
                "Scenario-only visualization auxiliary directory. Inventory only; "
                "never use it as a training/validation/test benchmark split."
            ),
        }
    raise KeyError(split)


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


def _representation_row(
    *,
    root: Path,
    representation: str,
    split: str,
    sample_scenario_shards: int,
) -> dict[str, object]:
    if representation == "scenario":
        spec = SCENARIO_SPLITS.get(split)
        if spec is None:
            return {
                "applicable": False,
                "released_for_this_representation": False,
                "reason": f"WOMD v1.3.1 layout used by COWP has no scenario/{split} contract",
            }
        glob = str(root / "uncompressed" / "scenario" / split / "*.tfrecord*")
        files = resolve_scenario_files(glob)
        return {
            "applicable": True,
            "released_for_this_representation": True,
            "glob": glob,
            "shards": _shard_manifest(files, expected=spec["expected_shards"]),
            "sample": _scenario_sample_stats(files, sample_scenario_shards) if files else {"sampled_records": 0},
        }
    if representation == "tf_example":
        spec = TFEXAMPLE_SPLITS.get(split)
        if spec is None:
            return {
                "applicable": False,
                "released_for_this_representation": False,
                "reason": (
                    f"WOMD v1.3.1 layout used by COWP has no tf_example/{split}; "
                    "do not synthesize or require this path"
                ),
            }
        glob = str(root / "uncompressed" / "tf_example" / split / "*.tfrecord*")
        files = resolve_tfexample_files(glob)
        return {
            "applicable": True,
            "released_for_this_representation": True,
            "glob": glob,
            "shards": _shard_manifest(files, expected=spec["expected_shards"]),
        }
    raise ValueError(representation)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit the representation-aware WOMD 1.3.1 split layout before COWP "
            "dataset construction. Scenario-only auxiliary splits are never "
            "mistaken for tf.Example splits."
        )
    )
    ap.add_argument("--womd-root", required=True)
    ap.add_argument("--sample-scenario-shards", type=int, default=16)
    ap.add_argument("--full-validation-interactive-overlap", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.womd_root)
    result: dict[str, object] = {
        "schema_version": "cowp_womd_v1_3_1_split_layout_v2",
        "womd_root": str(root.resolve()),
        "release_representation_contract": {
            "scenario_splits": list(SCENARIO_SPLITS),
            "tf_example_splits": list(TFEXAMPLE_SPLITS),
            "scenario_only_splits": sorted(set(SCENARIO_SPLITS) - set(TFEXAMPLE_SPLITS)),
            "tf_example_only_splits": sorted(set(TFEXAMPLE_SPLITS) - set(SCENARIO_SPLITS)),
            "explicitly_not_expected": [
                "uncompressed/tf_example/training_20s",
                "uncompressed/tf_example/visualization",
            ],
        },
        "splits": {},
        "benchmark_recommendation": {
            "primary_train": "scenario/training + matched tf_example/training",
            "primary_validation": "scenario/validation + matched tf_example/validation",
            "secondary_interaction_stress": (
                "scenario/validation_interactive + matched tf_example/validation_interactive, "
                "reported separately after overlap audit"
            ),
            "blind_physical_test": "testing / testing_interactive only through official/blind evaluation; no future-dependent mechanism GT",
            "excluded_from_current_pipeline": ["scenario/training_20s", "scenario/visualization"],
        },
    }

    split_rows: dict[str, object] = {}
    for split in ALL_SPLITS:
        split_rows[split] = {
            **_split_role(split),
            "scenario": _representation_row(
                root=root,
                representation="scenario",
                split=split,
                sample_scenario_shards=args.sample_scenario_shards,
            ),
            "tf_example": _representation_row(
                root=root,
                representation="tf_example",
                split=split,
                sample_scenario_shards=args.sample_scenario_shards,
            ),
        }
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
            result["validation_interactive_overlap"] = {
                "available": False,
                "reason": "standard validation and/or scenario/validation_interactive is absent locally",
            }

    # Only the 9-second primary train/validation pairs are hard requirements for
    # the current COWP rebuild. Optional challenge, blind-test, 20-second and
    # visualization directories are inventory only and cannot fail this gate.
    primary_checks: dict[str, bool] = {}
    for split in ("training", "validation"):
        scenario_row = split_rows[split]["scenario"]
        tf_row = split_rows[split]["tf_example"]
        primary_checks[f"scenario_{split}_complete"] = bool(scenario_row["shards"]["complete"])
        primary_checks[f"tf_example_{split}_complete"] = bool(tf_row["shards"]["complete"])
    result["primary_checks"] = primary_checks
    result["pass_primary_layout"] = all(primary_checks.values())
    result["interpretation"] = (
        "Primary 9-second training/validation Scenario+tf.Example layout is complete. "
        "Scenario-only training_20s/visualization are not expected to have tf.Example peers."
        if result["pass_primary_layout"]
        else
        "Primary 9-second training/validation layout is incomplete. Optional missing auxiliary splits are not part of this failure."
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["pass_primary_layout"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
