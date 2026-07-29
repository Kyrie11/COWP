from pathlib import Path

from ocrap.config.defaults import DEFAULT_CONFIG
from ocrap.data.build.builder import build_dataset


def test_artifact_builder_smoke(tmp_path: Path):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({"num_synthetic_scenarios": 1, "num_candidate_prefixes": 3, "num_reactive_futures": 1, "num_targeted_futures": 3, "max_times_per_scenario": 1, "max_biased_times_per_scenario": 0, "bev_resolution_m": 4.0})
    result = build_dataset(tmp_path / "ds", cfg)
    assert result["num_samples"] == 3
