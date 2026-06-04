from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from cowp.core.constants import ObjectType
from cowp.core.types import Lane, MapData, ScenarioData


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - external dependency path
        raise ImportError("TensorFlow is required to stream WOMD TFRecord files. Install tensorflow>=2.11.") from exc
    return tf


def _import_scenario_proto():
    try:
        from waymo_open_dataset.protos import scenario_pb2  # type: ignore
    except Exception as exc:  # pragma: no cover - external dependency path
        raise ImportError(
            "waymo-open-dataset is required to parse Scenario protos. "
            "Install the package variant matching your TensorFlow version, e.g. waymo-open-dataset-tf-2-12-0."
        ) from exc
    return scenario_pb2


def iter_scenario_records(patterns: str | list[str]) -> Iterator[bytes]:
    tf = _import_tensorflow()
    if isinstance(patterns, str):
        patterns = [patterns]
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No Scenario proto TFRecord files matched: {patterns}")
    for rec in tf.data.TFRecordDataset(files):
        yield bytes(rec.numpy())


def iter_scenarios(patterns: str | list[str]) -> Iterator[object]:
    scenario_pb2 = _import_scenario_proto()
    for raw in iter_scenario_records(patterns):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw)
        yield scenario


def _point_to_xyz(point) -> tuple[float, float, float]:
    return (float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0)), float(getattr(point, "z", 0.0)))


def _polyline(points) -> np.ndarray:
    return np.asarray([_point_to_xyz(p) for p in points], dtype=np.float32)


def _neighbor_ids(neighbors) -> tuple[int, ...]:
    ids = []
    for n in neighbors:
        for attr in ("feature_id", "self_start_index", "neighbor_start_index"):
            if hasattr(n, attr) and attr == "feature_id":
                ids.append(int(getattr(n, attr)))
                break
    return tuple(ids)


def parse_map_features(scenario) -> MapData:
    map_data = MapData()
    stop_controlled: set[int] = set()
    # First pass: stop signs can reference lane ids.
    for feature in scenario.map_features:
        fid = int(getattr(feature, "id", len(map_data.stop_signs)))
        if feature.HasField("stop_sign"):
            ss = feature.stop_sign
            lane_ids = tuple(int(x) for x in getattr(ss, "lane", []))
            stop_controlled.update(lane_ids)
            map_data.stop_signs[fid] = {"lane_ids": lane_ids, "position": np.asarray(_point_to_xyz(ss.position), dtype=np.float32) if ss.HasField("position") else np.zeros(3, dtype=np.float32)}
    for feature in scenario.map_features:
        fid = int(getattr(feature, "id", 0))
        if feature.HasField("lane"):
            lane_proto = feature.lane
            poly = _polyline(lane_proto.polyline)
            entry = tuple(int(x) for x in getattr(lane_proto, "entry_lanes", []))
            exit_ = tuple(int(x) for x in getattr(lane_proto, "exit_lanes", []))
            left = tuple(int(getattr(n, "feature_id", -1)) for n in getattr(lane_proto, "left_neighbors", []))
            right = tuple(int(getattr(n, "feature_id", -1)) for n in getattr(lane_proto, "right_neighbors", []))
            speed_limit = float(getattr(lane_proto, "speed_limit_mph", 31.0)) * 0.44704
            map_data.lanes[fid] = Lane(
                lane_id=fid,
                polyline=poly,
                speed_limit_mps=speed_limit if speed_limit > 0 else 13.9,
                turn_direction=int(getattr(lane_proto, "turn", 0)),
                entry_lanes=entry,
                exit_lanes=exit_,
                left_neighbors=tuple(x for x in left if x >= 0),
                right_neighbors=tuple(x for x in right if x >= 0),
                controlled_by_stop=fid in stop_controlled,
                controlled_by_signal=False,
                lane_type=int(getattr(lane_proto, "type", 0)),
            )
        elif feature.HasField("road_line"):
            map_data.road_lines[fid] = _polyline(feature.road_line.polyline)
        elif feature.HasField("road_edge"):
            map_data.road_edges[fid] = _polyline(feature.road_edge.polyline)
        elif feature.HasField("crosswalk"):
            map_data.crosswalks[fid] = _polyline(feature.crosswalk.polygon)
        elif feature.HasField("speed_bump"):
            map_data.speed_bumps[fid] = _polyline(feature.speed_bump.polygon)
    for t, dyn in enumerate(scenario.dynamic_map_states):
        state = {"t": t, "lane_states": []}
        for lane_state in getattr(dyn, "lane_states", []):
            lane_ids = tuple(int(x) for x in getattr(lane_state, "lane", []))
            if not lane_ids and hasattr(lane_state, "lane_id"):
                lane_ids = (int(lane_state.lane_id),)
            for lid in lane_ids:
                if lid in map_data.lanes:
                    map_data.lanes[lid].controlled_by_signal = True
            state["lane_states"].append({"lane_ids": lane_ids, "state": int(getattr(lane_state, "state", 0))})
        map_data.dynamic_signals.append(state)
    return map_data


def scenario_to_scene(scenario, keep_raw: bool = False) -> ScenarioData:
    timestamps = np.asarray(list(scenario.timestamps_seconds), dtype=np.float32)
    n = len(scenario.tracks)
    t = len(timestamps)
    states = np.zeros((n, t, 11), dtype=np.float32)
    object_type = np.zeros(n, dtype=np.int32)
    track_id = np.zeros(n, dtype=np.int64)
    for i, track in enumerate(scenario.tracks):
        object_type[i] = int(getattr(track, "object_type", ObjectType.UNKNOWN))
        track_id[i] = int(getattr(track, "id", i))
        for j, st in enumerate(track.states[:t]):
            vx = float(getattr(st, "velocity_x", 0.0))
            vy = float(getattr(st, "velocity_y", 0.0))
            states[i, j] = np.asarray(
                [
                    float(getattr(st, "center_x", 0.0)),
                    float(getattr(st, "center_y", 0.0)),
                    float(getattr(st, "center_z", 0.0)),
                    vx,
                    vy,
                    float(np.hypot(vx, vy)),
                    float(getattr(st, "heading", 0.0)),
                    float(getattr(st, "length", 0.0)),
                    float(getattr(st, "width", 0.0)),
                    float(getattr(st, "height", 0.0)),
                    1.0 if bool(getattr(st, "valid", False)) else 0.0,
                ],
                dtype=np.float32,
            )
    tracks_to_predict = []
    for r in getattr(scenario, "tracks_to_predict", []):
        tracks_to_predict.append(int(getattr(r, "track_index", -1)))
    return ScenarioData(
        scenario_id=str(scenario.scenario_id),
        timestamps=timestamps,
        current_time_index=int(scenario.current_time_index),
        states=states,
        object_type=object_type,
        track_id=track_id,
        sdc_track_index=int(scenario.sdc_track_index),
        objects_of_interest=np.asarray(list(getattr(scenario, "objects_of_interest", [])), dtype=np.int64),
        tracks_to_predict=np.asarray(tracks_to_predict, dtype=np.int32),
        map_data=parse_map_features(scenario),
        raw=scenario if keep_raw else None,
    )


def write_index(patterns: str | list[str], output_jsonl: str | Path, limit: int | None = None) -> None:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for scenario in iter_scenarios(patterns):
            scene = scenario_to_scene(scenario, keep_raw=False)
            item = {
                "scenario_id": scene.scenario_id,
                "num_agents": scene.num_agents,
                "num_steps": scene.num_steps,
                "current_time_index": scene.current_time_index,
                "sdc_track_index": scene.sdc_track_index,
                "objects_of_interest": scene.objects_of_interest.tolist(),
                "tracks_to_predict": scene.tracks_to_predict.tolist(),
                "num_lanes": len(scene.map_data.lanes),
            }
            f.write(json.dumps(item) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
