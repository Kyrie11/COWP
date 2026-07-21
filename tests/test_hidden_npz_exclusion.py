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
