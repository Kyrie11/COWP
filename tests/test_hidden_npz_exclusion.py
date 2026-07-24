from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset


def test_dataset_excludes_hidden_npz_metadata(tmp_path: Path) -> None:
    np.savez(tmp_path / "scenario_001.npz", feature=np.asarray([1], dtype=np.float32))
    np.savez(
        tmp_path / ".cowp_sampler_weights_pw3_sw2_v2.npz",
        weights=np.asarray([1.0], dtype=np.float64),
        metadata=np.asarray("{}"),
    )

    ds = COWPNpzDataset(tmp_path)
    assert [p.name for p in ds.paths] == ["scenario_001.npz"]


def test_cache_sufficiency_scan_excludes_hidden_npz(tmp_path: Path) -> None:
    import importlib

    np.savez(tmp_path / "scenario_001.npz", feature=np.asarray([1], dtype=np.float32))
    np.savez(tmp_path / ".cowp_sampler_weights_pw3_sw2_v2.npz", weights=np.asarray([1.0]))

    module = importlib.import_module("cowp.scripts.19_diagnose_waymax_cache_sufficiency")
    assert [p.name for p in module._scenario_paths(tmp_path)] == ["scenario_001.npz"]
