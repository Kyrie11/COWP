from pathlib import Path

from ocrap.config.defaults import DEFAULT_CONFIG
from ocrap.data.build.builder import build_dataset
from ocrap.data.build.papercheck import papercheck_dataset


def test_papercheck_passes_artifact_fixture(tmp_path: Path):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({"num_synthetic_scenarios": 1, "num_candidate_prefixes": 4, "num_reactive_futures": 2, "num_targeted_futures": 4, "max_times_per_scenario": 1, "max_biased_times_per_scenario": 0, "bev_resolution_m": 4.0})
    ds = tmp_path / "artifact_fixture"
    build_dataset(ds, cfg)
    report = papercheck_dataset(ds)
    assert report["failures"] == []
    assert report["artifact_fraction"] > 0
    assert report["hidden_emergence_count"] == report["hidden_from_unknown_count"]
