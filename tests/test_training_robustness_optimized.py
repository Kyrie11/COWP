from __future__ import annotations

import numpy as np
import torch

from cowp.models.losses import natural_loss, witness_loss
from cowp.data.dataset import TorchCOWPDataset


def test_natural_loss_clamps_non_binary_priority_and_bad_source_labels():
    B, A, M, T, D = 1, 1, 4, 3, 7
    pred = {
        "traj": torch.zeros(B, A, M, T, D),
        "logits": torch.zeros(B, A, M),
        "source_logits": torch.zeros(B, A, M, 4),
        "priority_logits": torch.zeros(B, A, M),
    }
    batch = {
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/natural/valid": torch.ones(B, A, M, dtype=torch.bool),
        "cowp/natural/traj": torch.zeros(B, A, M, T, D),
        "cowp/natural/weight": torch.ones(B, A, M) / M,
        # Includes invalid values that used to be dangerous for CUDA scatter kernels.
        "cowp/natural/source": torch.tensor([[[0, 1, 999, -5]]]),
        # Includes values outside [0,1] that used to trigger CUDA BCE asserts.
        "cowp/natural/priority_preserved": torch.tensor([[[1.0, 2.0, -3.0, float("nan")]]]),
    }
    losses = natural_loss(pred, batch, {})
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["priority"])


def test_witness_loss_clamps_bad_token_labels():
    B, K, A = 1, 2, 1
    pred = {
        "exist_logits": torch.zeros(B, K, A),
        "token_logits": torch.zeros(B, K, A, 7),
        "burden_total": torch.zeros(B, K, A),
        "burden_components": torch.zeros(B, K, A, 6),
        "conflict_interval": torch.zeros(B, K, A, 2),
        "opr": torch.ones(B, K, A),
        "c_i": torch.zeros(B, K, A),
    }
    batch = {
        "cowp/candidates/valid": torch.ones(B, K, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/witness/exists": torch.ones(B, K, A, dtype=torch.bool),
        "cowp/witness/token": torch.tensor([[[999], [-10]]]),
        "cowp/witness/burden_total": torch.zeros(B, K, A),
        "cowp/witness/burden_components": torch.zeros(B, K, A, 6),
        "cowp/witness/conflict_interval": torch.zeros(B, K, A, 2),
        "cowp/witness/opr": torch.ones(B, K, A),
        "cowp/witness/c_i": torch.zeros(B, K, A),
    }
    losses = witness_loss(pred, batch, {})
    assert torch.isfinite(losses["loss"])


def test_stage_dataset_does_not_load_unneeded_heavy_arrays(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez(
        path,
        **{
            "cowp__critical__track_index": np.array([0], dtype=np.int64),
            "cowp__critical__valid": np.array([True]),
            "cowp__natural__traj": np.zeros((1, 2, 3, 7), dtype=np.float32),
            "cowp__natural__valid": np.ones((1, 2), dtype=bool),
            "cowp__natural__weight": np.ones((1, 2), dtype=np.float32) / 2,
            "cowp__natural__source": np.zeros((1, 2), dtype=np.int64),
            "cowp__natural__priority_preserved": np.ones((1, 2), dtype=bool),
            "cowp__response__traj": np.zeros((4, 1, 32, 80, 7), dtype=np.float32),
            "state__history": np.zeros((1, 11, 11), dtype=np.float32),
            "state__agent_valid": np.array([True]),
            "map__conflict_regions": np.zeros((1, 8), dtype=np.float32),
            "map__conflict_region_valid": np.array([False]),
        },
    )
    ds = TorchCOWPDataset(tmp_path, stage="representation")
    sample = ds[0]
    assert "cowp/natural/traj" in sample
    assert "cowp/response/traj" not in sample
