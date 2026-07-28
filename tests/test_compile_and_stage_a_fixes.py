from __future__ import annotations

import importlib

import numpy as np
import torch

from cowp.data.dataset import TorchCOWPDataset
from cowp.models.graph_encoder import GraphEncoder


def test_history_mean_uses_tensor_where_without_data_dependent_branch():
    hist = torch.zeros(2, 3, 4, 11)
    hist[0, 0, :, 0] = torch.tensor([0.0, 2.0, 4.0, 8.0])
    hist[0, 0, :, 10] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    # Empty history for [1, 2] should fall back to clean arithmetic mean rather
    # than entering Python-side tensor control flow.
    hist[1, 2, :, 0] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    out = GraphEncoder._history_mean(hist)
    assert torch.allclose(out[0, 0, 0], torch.tensor(3.0))
    assert torch.allclose(out[1, 2, 0], torch.tensor(4.0))
    assert torch.isfinite(out).all()


def test_stage_a_dataset_loads_only_specific_map_keys(tmp_path):
    np.savez(
        tmp_path / "sample.npz",
        **{
            "cowp__critical__track_index": np.array([0], dtype=np.int64),
            "cowp__critical__valid": np.array([True]),
            "cowp__natural__traj": np.zeros((1, 2, 3, 7), dtype=np.float32),
            "cowp__natural__valid": np.ones((1, 2), dtype=bool),
            "cowp__natural__weight": np.ones((1, 2), dtype=np.float32) / 2,
            "cowp__natural__source": np.zeros((1, 2), dtype=np.int64),
            "cowp__natural__priority_preserved": np.ones((1, 2), dtype=bool),
            "state__history": np.zeros((1, 11, 11), dtype=np.float32),
            "state__agent_valid": np.array([True]),
            "map__conflict_regions": np.zeros((4, 8), dtype=np.float32),
            "map__conflict_region_valid": np.zeros((4,), dtype=bool),
            "map__very_large_unused_aux": np.zeros((1024, 1024), dtype=np.float32),
        },
    )
    sample = TorchCOWPDataset(tmp_path, stage="representation")[0]
    assert "map/conflict_regions" in sample
    assert "map/conflict_region_valid" in sample
    assert "map/very_large_unused_aux" not in sample


def test_checkpoint_helpers_strip_compiled_prefix():
    train_mod = importlib.import_module("cowp.scripts.03_train")
    model = torch.nn.Linear(2, 1)
    state = {f"_orig_mod.{k}": v.clone() for k, v in model.state_dict().items()}
    fresh = torch.nn.Linear(2, 1)
    train_mod._load_model_state_robust(fresh, state)
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)
