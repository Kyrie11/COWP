from __future__ import annotations

import torch

from cowp.data.womd_features import build_agent_history_from_womd, has_womd_state


def test_womd_flat_tensor_cache_features_become_state_history():
    B, N, Tp = 2, 3, 10
    batch = {
        "womd/state/past/x": torch.arange(B * N * Tp, dtype=torch.float32).reshape(B, N * Tp),
        "womd/state/past/y": torch.zeros(B, N * Tp),
        "womd/state/past/valid": torch.ones(B, N * Tp),
        "womd/state/current/x": torch.ones(B, N),
        "womd/state/current/y": torch.ones(B, N) * 2,
        "womd/state/current/valid": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
        "womd/state/is_sdc": torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32),
    }
    assert has_womd_state(batch)
    hist, mask = build_agent_history_from_womd(batch, max_agents=N, history_steps=11, d_state=11)
    assert hist.shape == (B, N, 11, 11)
    assert mask.tolist() == [[True, True, False], [True, False, False]]
    assert hist[0, 0, 0, 0].item() == 0.0
    assert hist[0, 0, -1, 0].item() == 1.0
    assert hist[0, 1, -1, 1].item() == 2.0
