from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cowp.data.build_cache import _sdc_path_contract_errors, _waymax_missing_required_womd_keys
from cowp.data.parse_scenario_proto import _import_scenario_proto, resolve_glob_patterns as resolve_scenario_files
from cowp.data.parse_tfexample import decode_parsed_tfexample, iter_tfexample_records_by_file, parse_tfexample, resolve_glob_patterns as resolve_tfexample_files, scenario_id_from_parsed_tfexample


def _sample_files(files: list[str], count: int) -> list[str]:
    if not files:
        return []
    n = min(max(int(count), 1), len(files))
    idx = sorted(set(np.linspace(0, len(files) - 1, num=n, dtype=np.int64).tolist()))
    return [files[i] for i in idx]


def _first_tfexample(filename: str):
    for _f, _idx, raw in iter_tfexample_records_by_file([filename]):
        return parse_tfexample(raw)
    raise RuntimeError(f"empty TFRecord shard: {filename}")


def _audit_tfexample_split(name: str, glob_pattern: str, samples: int, require_sdc_paths: bool) -> dict:
    files = resolve_tfexample_files(glob_pattern)
    selected = _sample_files(files, samples)
    errors: list[dict[str, object]] = []
    scenario_ids: list[str] = []
    path_ready = 0
    core_ready = 0
    for filename in selected:
        try:
            parsed = _first_tfexample(filename)
            sid = scenario_id_from_parsed_tfexample(parsed)
            ex = decode_parsed_tfexample(parsed)
            scenario_ids.append(sid)
            local: list[str] = []
            core_missing = _waymax_missing_required_womd_keys(ex)
            if core_missing:
                local.extend(f"core_missing:{x}" for x in core_missing)
            else:
                core_ready += 1

            ids = np.asarray(ex.get("state/id", [])).reshape(-1)
            n_obj = int(ids.size)
            if n_obj != 128:
                local.append(f"state/id size={n_obj}, expected=128")
            is_sdc = np.asarray(ex.get("state/is_sdc", [])).reshape(-1)
            if is_sdc.size != n_obj or int(np.sum(is_sdc > 0)) != 1:
                local.append(f"state/is_sdc size={is_sdc.size}, positives={int(np.sum(is_sdc > 0))}, expected one of {n_obj}")
            for prefix, steps in (("past", 10), ("current", 1), ("future", 80)):
                for field in ("x", "y", "valid"):
                    arr = np.asarray(ex.get(f"state/{prefix}/{field}", [])).reshape(-1)
                    expected = n_obj * steps
                    if arr.size != expected:
                        local.append(f"state/{prefix}/{field} size={arr.size}, expected={expected}")
            if n_obj and is_sdc.size == n_obj:
                sdc_idx = int(np.argmax(is_sdc))
                cur_valid = np.asarray(ex.get("state/current/valid", [])).reshape(-1)
                if cur_valid.size == n_obj and not bool(cur_valid[sdc_idx] > 0):
                    local.append("SDC current state is invalid")

            road_xyz = np.asarray(ex.get("roadgraph_samples/xyz", [])).reshape(-1)
            road_valid = np.asarray(ex.get("roadgraph_samples/valid", [])).reshape(-1)
            if road_xyz.size and road_valid.size and road_xyz.size != road_valid.size * 3:
                local.append(f"roadgraph xyz size={road_xyz.size}, expected={road_valid.size * 3}")
            path_errors = _sdc_path_contract_errors(ex)
            if not path_errors:
                path_ready += 1
            elif require_sdc_paths:
                local.extend(f"sdc_path:{x}" for x in path_errors)
            if local:
                errors.append({"file": filename, "scenario_id": sid, "errors": local[:32]})
        except Exception as exc:
            errors.append({"file": filename, "scenario_id": None, "errors": [repr(exc)]})
    return {
        "split": name,
        "glob": glob_pattern,
        "num_shards": len(files),
        "sampled_shards": len(selected),
        "core_waymax_ready_samples": core_ready,
        "sdc_paths_ready_samples": path_ready,
        "unique_sampled_scenario_ids": len(set(scenario_ids)),
        "errors": errors[:50],
        "pass": not errors,
    }


def _audit_scenario_split(name: str, glob_pattern: str, samples: int) -> dict:
    files = resolve_scenario_files(glob_pattern)
    selected = _sample_files(files, samples)
    scenario_pb2 = _import_scenario_proto()
    # TensorFlow is imported lazily by the parser module helper via TFRecordDataset.
    from cowp.data.parse_scenario_proto import _import_tensorflow
    tf = _import_tensorflow()
    errors: list[dict[str, object]] = []
    for filename in selected:
        try:
            ds = tf.data.TFRecordDataset([filename]).take(1)
            raw = next(iter(ds), None)
            if raw is None:
                raise RuntimeError("empty TFRecord shard")
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(bytes(raw.numpy()))
            local: list[str] = []
            if len(sc.timestamps_seconds) != 91:
                local.append(f"timestamps={len(sc.timestamps_seconds)}, expected=91")
            if int(sc.current_time_index) != 10:
                local.append(f"current_time_index={int(sc.current_time_index)}, expected=10")
            if not (0 <= int(sc.sdc_track_index) < len(sc.tracks)):
                local.append(f"invalid sdc_track_index={int(sc.sdc_track_index)} for tracks={len(sc.tracks)}")
            else:
                sdc = sc.tracks[int(sc.sdc_track_index)]
                if len(sdc.states) != len(sc.timestamps_seconds):
                    local.append(f"SDC states={len(sdc.states)}, timestamps={len(sc.timestamps_seconds)}")
                elif not bool(sdc.states[int(sc.current_time_index)].valid):
                    local.append("SDC current Scenario state is invalid")
            if local:
                errors.append({"file": filename, "scenario_id": str(sc.scenario_id), "errors": local})
        except Exception as exc:
            errors.append({"file": filename, "scenario_id": None, "errors": [repr(exc)]})
    return {
        "split": name,
        "glob": glob_pattern,
        "num_shards": len(files),
        "sampled_shards": len(selected),
        "errors": errors[:50],
        "pass": not errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-fast WOMD 1.3.1 contract preflight before expensive COWP label rebuilds.")
    ap.add_argument("--tfexample-train-glob", required=True)
    ap.add_argument("--tfexample-val-glob", required=True)
    ap.add_argument("--scenario-train-glob", default=None)
    ap.add_argument("--scenario-val-glob", default=None)
    ap.add_argument("--sample-shards", type=int, default=64)
    ap.add_argument("--scenario-sample-shards", type=int, default=32)
    ap.add_argument("--require-sdc-paths", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    results = {
        "tfexample_train": _audit_tfexample_split("training", args.tfexample_train_glob, args.sample_shards, args.require_sdc_paths),
        "tfexample_val": _audit_tfexample_split("validation", args.tfexample_val_glob, args.sample_shards, args.require_sdc_paths),
    }
    if args.scenario_train_glob:
        results["scenario_train"] = _audit_scenario_split("training", args.scenario_train_glob, args.scenario_sample_shards)
    if args.scenario_val_glob:
        results["scenario_val"] = _audit_scenario_split("validation", args.scenario_val_glob, args.scenario_sample_shards)
    passed = all(bool(x.get("pass", False)) for x in results.values())
    report = {
        "schema_version": "cowp_womd_v1_3_1_preflight_v1",
        "pass": bool(passed),
        "require_sdc_paths": bool(args.require_sdc_paths),
        "results": results,
        "interpretation": (
            "Sampled Scenario/tf.Example shards satisfy the expected 1s-history + current + 8s-future contract; full cache construction must still enforce the per-matched-scene SDC-path contract."
            if passed else
            "WOMD preflight failed. Do not spend time on the COWP full label rebuild until split/version/path semantics are fixed."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
