from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from ocrap.data.schema import RawScenario

from .agent_selection import select_sdc_first_indices
from .dynamic_map import parse_dynamic_map
from .feature_extraction import extract_agent_arrays
from .map_features import parse_map_features
from .route import parse_route
from .torch_tfrecord import TFRecordReader, expand_paths


def _load_scenario_pb2():
    try:
        from waymo_open_dataset.protos import scenario_pb2  # type: ignore
        return scenario_pb2
    except Exception as e:
        raise ImportError("WOMD Scenario proto definitions are required to parse bytes. Install a package providing waymo_open_dataset.protos.scenario_pb2 or generate the proto module; TensorFlow is not required by this reader.") from e


def parse_scenario_bytes(data: bytes):
    scenario_pb2 = _load_scenario_pb2()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(data)
    return scenario


def parse_scenario_proto(scenario, max_agents: int = 64, max_polylines: int = 256, max_points: int = 64, max_dynamic_signals: int = 16) -> RawScenario:
    tracks = list(getattr(scenario, "tracks", []))
    T = max((len(getattr(tr, "states", [])) for tr in tracks), default=0)
    indices = select_sdc_first_indices(scenario, max_agents=max_agents)
    states, valid, object_ids = extract_agent_arrays(scenario, indices, max_agents)
    timestamps = np.asarray(list(getattr(scenario, "timestamps_seconds", [])), dtype=np.float32)
    if timestamps.size < T:
        timestamps = np.arange(T, dtype=np.float32) * 0.1
    maps, map_valid = parse_map_features(scenario, max_polylines=max_polylines, max_points=max_points)
    route = parse_route(scenario, maps, map_valid, max_points=max_points)
    dyn = parse_dynamic_map(scenario, T=T, max_signals=max_dynamic_signals)
    original_sdc = int(getattr(scenario, "sdc_track_index", 0))
    agent_index_map = {int(new): int(old) for new, old in enumerate(indices)}
    # SDC-first output contract.
    return RawScenario(
        scenario_id=str(getattr(scenario, "scenario_id", "")),
        timestamps=timestamps[:T],
        sdc_track_index=0,
        agent_states=states,
        agent_valid=valid,
        map_polylines=maps,
        map_valid=map_valid,
        route=route,
        dynamic_map=dyn,
        object_ids=object_ids,
        metadata={"original_scenario_id": str(getattr(scenario, "scenario_id", "")), "original_sdc_track_index": original_sdc, "agent_index_map": agent_index_map, "source": "womd"},
    )


class WOMDScenarioIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, patterns, max_scenarios: int | None = None, parser_cfg: dict | None = None, verify_crc: bool = True):
        super().__init__()
        self.paths = expand_paths(patterns)
        self.max_scenarios = max_scenarios
        self.parser_cfg = parser_cfg or {}
        self.verify_crc = verify_crc

    def __iter__(self) -> Iterator[RawScenario]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            start, stride = 0, 1
        else:
            start, stride = info.id, info.num_workers
        count = 0
        reader = TFRecordReader(self.paths, verify_crc=self.verify_crc, start_shard=start, shard_stride=stride)
        for rec_idx, rec in enumerate(reader):
            try:
                scenario = parse_scenario_bytes(rec.data)
                raw = parse_scenario_proto(
                    scenario,
                    max_agents=int(self.parser_cfg.get("max_agents", 64)),
                    max_polylines=int(self.parser_cfg.get("max_map_polylines", 256)),
                    max_points=int(self.parser_cfg.get("max_polyline_points", 64)),
                    max_dynamic_signals=int(self.parser_cfg.get("max_dynamic_signals", 16)),
                )
                raw.metadata.update({"tfrecord_path": rec.path, "tfrecord_offset": rec.offset, "scenario_record_index": rec_idx})
            except Exception as e:
                raise RuntimeError(f"Failed to parse WOMD scenario at path={rec.path} offset={rec.offset} index={rec_idx}: {e}") from e
            yield raw
            count += 1
            if self.max_scenarios is not None and count >= int(self.max_scenarios):
                return


def iter_womd_scenarios(patterns, max_scenarios: int | None = None, parser_cfg: dict | None = None, verify_crc: bool = True) -> Iterator[RawScenario]:
    yield from WOMDScenarioIterableDataset(patterns, max_scenarios=max_scenarios, parser_cfg=parser_cfg, verify_crc=verify_crc)
