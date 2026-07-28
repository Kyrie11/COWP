from __future__ import annotations

import numpy as np
import torch

from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel


def _sample(path, *, with_state=True, womd_prefix=True, with_natural=True, x0=10.0):
    data = {
        "cowp__critical__track_index": np.array([0], dtype=np.int64),
        "cowp__critical__track_id": np.array([123], dtype=np.int64),
        "cowp__critical__valid": np.array([True]),
        "map__conflict_regions": np.zeros((2, 8), dtype=np.float32),
        "map__conflict_region_valid": np.zeros((2,), dtype=bool),
    }
    if with_natural:
        data.update({
            "cowp__natural__traj": np.zeros((1, 2, 3, 7), dtype=np.float32),
            "cowp__natural__valid": np.ones((1, 2), dtype=bool),
            "cowp__natural__weight": np.ones((1, 2), dtype=np.float32) / 2,
            "cowp__natural__source": np.zeros((1, 2), dtype=np.int64),
            "cowp__natural__priority_preserved": np.ones((1, 2), dtype=bool),
        })
    if with_state:
        pre = "womd__state" if womd_prefix else "state"
        hist = np.zeros((1, 11, 11), dtype=np.float32)
        hist[0, -1, 0] = x0
        hist[0, -1, 1] = -5.0
        hist[0, -1, 3] = 4.8
        hist[0, -1, 4] = 1.9
        hist[0, -1, 6] = 0.2
        hist[0, -1, 10] = 1.0
        data[f"{pre}__history"] = hist
        data[f"{pre}__agent_valid"] = np.array([True])
        data[f"{pre}__id"] = np.array([123], dtype=np.int64)
    np.savez(path, **data)


def test_dataset_canonicalizes_womd_state_and_skips_partial_stage_a_sample(tmp_path):
    _sample(tmp_path / "000_bad.npz", with_state=False, with_natural=False)
    _sample(tmp_path / "001_good.npz", with_state=True, womd_prefix=True, with_natural=True)
    ds = TorchCOWPDataset(tmp_path, stage="representation")
    item = ds[0]
    assert "state/history" in item
    assert "womd/state/history" not in item
    assert "cowp/natural/traj" in item
    batch = collate_torch([item, ds[1]])
    assert "state/history" in batch
    assert "cowp/natural/traj" in batch


def test_model_anchors_natural_trajectory_to_current_critical_state():
    cfg = {"model": {"d_state": 11, "d_model": 16, "num_heads": 4, "num_layers": 1, "max_agents": 2, "max_natural_alternatives": 2, "future_steps": 3}}
    model = COWPModel(cfg)
    hist = torch.zeros(1, 2, 11, 11)
    hist[:, :, :, 10] = 1.0
    hist[0, 1, -1, 0] = 123.0
    hist[0, 1, -1, 1] = -7.0
    hist[0, 1, -1, 3] = 4.5
    hist[0, 1, -1, 4] = 2.0
    batch = {
        "state/history": hist,
        "state/agent_valid": torch.tensor([[True, True]]),
        "cowp/critical/track_index": torch.tensor([[1]]),
        "cowp/critical/valid": torch.tensor([[True]]),
    }
    out = model(batch, stage="representation")
    xy0 = out["natural"]["traj"][0, 0, :, 0, :2]
    assert torch.all(torch.abs(xy0[:, 0] - 123.0) < 10.0)
    assert torch.all(torch.abs(xy0[:, 1] + 7.0) < 10.0)
