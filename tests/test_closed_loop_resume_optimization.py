from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ocrap.models.data import sample_to_feature, samples_to_feature_matrix
from ocrap.simulation import closed_loop_runner as clr


def _feature_sample(candidate_index: int) -> dict:
    rng = np.random.default_rng(123)
    sample = {
        "time_index": np.int64(17),
        "candidate_index": np.int64(candidate_index),
        "is_nominal": np.int64(candidate_index == 0),
        "ego_state": rng.normal(size=(9,)).astype(np.float32),
        "prefix_param": (rng.normal(size=(5,)) + candidate_index).astype(np.float32),
        "prefix_macro_id": np.int64(candidate_index),
        "prefix_macro_type_id": np.int64(candidate_index % 4),
        "utility": np.float32(2.5 - 0.1 * candidate_index),
        "hard_violation": np.float32(0.2 * candidate_index),
        "harm_proxy": np.float32(0.05 * candidate_index),
        "feasible": np.int64(1),
        "prefix_states": (rng.normal(size=(10, 8)) + candidate_index).astype(np.float32),
        "prefix_controls": (rng.normal(size=(10, 4)) + candidate_index).astype(np.float32),
        "agent_history": rng.normal(size=(11, 9, 16)).astype(np.float32),
        "agent_valid": np.ones((11, 9), dtype=np.float32),
        "bev_occ": rng.normal(size=(7, 32, 32)).astype(np.float32),
        "route": rng.normal(size=(20, 2)).astype(np.float32),
        "map_polylines": rng.normal(size=(8, 12, 4)).astype(np.float32),
        "dynamic_map": rng.normal(size=(11, 6, 4)).astype(np.float32),
    }
    return sample


def test_shared_scene_feature_matrix_is_exactly_equal() -> None:
    samples = [_feature_sample(i) for i in range(6)]
    # Candidate builders share these arrays at a replan. Make that identity
    # explicit so the optimized path is tested under its actual precondition.
    for key in ("agent_history", "agent_valid", "bev_occ", "route", "map_polylines", "dynamic_map", "ego_state"):
        for sample in samples[1:]:
            sample[key] = samples[0][key]
    expected = np.stack([sample_to_feature(sample, {}) for sample in samples], axis=0)
    actual = samples_to_feature_matrix(samples, {}, shared_scene=True)
    np.testing.assert_array_equal(actual, expected)


def _fake_scene_result(scene_id: str, rank: int) -> dict:
    return {
        "scene_id": scene_id,
        "bucket_name": None,
        "target_key": None,
        "target_time_index": None,
        "num_decisions": 0,
        "num_metric_steps": 0,
        "method": "nominal",
        "label_mode": "fast",
        "rank": rank,
        "metric_summary": {},
        "macro_counts": {},
        "selection_reason_counts": {},
    }


def test_closed_loop_resumes_completed_scenes_from_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raws = [SimpleNamespace(scenario_id=f"s{i}", metadata={}) for i in range(4)]
    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios", lambda *args, **kwargs: iter(raws))

    first_calls: list[str] = []

    def interrupted_rollout(raw, rank, *args, **kwargs):
        first_calls.append(raw.scenario_id)
        if raw.scenario_id == "s2":
            raise RuntimeError("simulated interruption")
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", interrupted_rollout)
    output = tmp_path / "closed.json"
    cfg = {
        "closed_loop": {
            "max_scenarios": 4,
            "method": "nominal",
            "resume": True,
            "save_partial": True,
            "partial_write_every_scenes": 4,
            "progress": False,
        },
        "selection": {"gamma_rec": 0.0},
        "waymax": {},
        "artifact": {},
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert first_calls == ["s0", "s1", "s2"]
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")
    assert journal.exists()
    assert len(journal.read_text().strip().splitlines()) == 2

    resumed_calls: list[str] = []

    def resumed_rollout(raw, rank, *args, **kwargs):
        resumed_calls.append(raw.scenario_id)
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", resumed_rollout)
    result = clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert resumed_calls == ["s2", "s3"]
    assert result["num_scenes"] == 4
    assert result["resume"]["resumed_rollouts"] == 2
    assert json.loads(output.read_text())["num_scenes"] == 4
    progress = json.loads(output.with_suffix(output.suffix + ".progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["completed_rollouts"] == 4


def test_legacy_partial_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raws = [SimpleNamespace(scenario_id=f"s{i}", metadata={}) for i in range(3)]
    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios", lambda *args, **kwargs: iter(raws))
    output = tmp_path / "legacy.json"
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps({
        "method": "nominal",
        "bucket_dataset": None,
        "raw_scenarios_seen": 2,
        "scenes": [_fake_scene_result("s0", 0), _fake_scene_result("s1", 1)],
    }))
    calls: list[str] = []

    def rollout(raw, rank, *args, **kwargs):
        calls.append(raw.scenario_id)
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", rollout)
    cfg = {
        "closed_loop": {"max_scenarios": 3, "method": "nominal", "resume": True, "progress": False},
        "selection": {"gamma_rec": 0.0},
        "waymax": {},
        "artifact": {},
    }
    result = clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert calls == ["s2"]
    assert result["num_scenes"] == 3
    assert result["resume"]["legacy_sources"] == ["partial"]
