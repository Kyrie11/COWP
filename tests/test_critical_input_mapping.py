import numpy as np
import torch

from cowp.data.dataset import align_critical_agents_to_womd_input, mask_out_of_range_critical_agents, TorchCOWPDataset
from cowp.models.cowp_model import COWPModel


def test_critical_track_id_maps_to_womd_input_index_and_preserves_valid():
    d = {
        "womd/state/id": np.array([101, 555, 303], dtype=np.int64),
        "womd/state/current/x": np.zeros(3, dtype=np.float32),
        "womd/state/current/valid": np.array([1, 1, 1], dtype=np.int64),
        "cowp/critical/track_index": np.array([149, 7], dtype=np.int32),
        "cowp/critical/track_id": np.array([555, 999], dtype=np.int64),
        "cowp/critical/valid": np.array([True, True]),
        "cowp/natural/valid": np.ones((2, 1), dtype=bool),
        "cowp/natural/weight": np.ones((2, 1), dtype=np.float32),
    }
    align_critical_agents_to_womd_input(d)
    mask_out_of_range_critical_agents(d)
    assert d["cowp/critical/input_index"].tolist() == [1, -1]
    assert d["cowp/critical/valid"].tolist() == [True, False]
    assert d["cowp/natural/valid"].tolist() == [[True], [False]]


def test_model_prefers_critical_input_index_over_scenario_track_index():
    B, N, H, D = 1, 3, 11, 11
    K, A, M, T = 2, 1, 2, 4
    hist = torch.zeros(B, N, H, D)
    hist[..., 10] = 1.0
    batch = {
        "state/history": hist,
        "state/agent_valid": torch.ones(B, N, dtype=torch.bool),
        "cowp/critical/track_index": torch.tensor([[149]]),
        "cowp/critical/input_index": torch.tensor([[1]]),
        "cowp/critical/valid": torch.ones(B, A, dtype=torch.bool),
        "cowp/natural/traj": torch.zeros(B, A, M, T, 7),
        "cowp/natural/weight": torch.ones(B, A, M) / M,
        "cowp/natural/source": torch.zeros(B, A, M, dtype=torch.long),
        "cowp/natural/valid": torch.ones(B, A, M, dtype=torch.bool),
        "cowp/natural/priority_preserved": torch.ones(B, A, M, dtype=torch.bool),
    }
    cfg = {"model": {"d_state": 11, "history_steps": 11, "d_model": 32, "num_heads": 4, "num_layers": 1, "dropout": 0.0, "max_agents": 3, "max_natural_alternatives": M, "future_steps": T, "max_safe_responses": 2, "token_count": 7}, "ablation": {}}
    out = COWPModel(cfg)(batch, stage="representation")
    assert out["critical_idx"].tolist() == [[1]]
    assert out["critical_mask"].tolist() == [[True]]


def test_existing_aligned_input_index_is_preserved_when_ids_are_unavailable():
    d = {
        "state/current/x": np.zeros(5, dtype=np.float32),
        "state/current/valid": np.ones(5, dtype=np.int64),
        "cowp/critical/track_index": np.array([149, 7], dtype=np.int64),
        "cowp/critical/input_index": np.array([1, 3], dtype=np.int64),
        "cowp/critical/valid": np.array([True, True]),
    }
    align_critical_agents_to_womd_input(d)
    assert d["cowp/critical/input_index"].tolist() == [1, 3]
