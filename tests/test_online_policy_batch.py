from __future__ import annotations

import numpy as np

from cowp.waymax_eval.policy_wrapper import build_online_batch


def _toy_state() -> np.ndarray:
    s = np.zeros((5, 11), dtype=np.float32)
    s[:, 7] = 4.8
    s[:, 8] = 1.9
    s[:, 9] = 1.6
    s[:, 10] = 1.0
    s[0, 0:6] = [0.0, 0.0, 0.0, 5.0, 0.0, 5.0]
    s[0, 6] = 0.0
    s[1, 0:6] = [16.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    s[2, 0:6] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    return s


def test_online_batch_has_nonzero_conflict_tokens_and_masks():
    cfg = {
        "limits": {"max_candidates": 16, "max_critical_agents": 4, "max_natural_alternatives": 4, "max_safe_responses": 4, "max_conflict_regions": 8, "max_agents": 8},
        "time": {"future_steps": 10, "dt": 0.1},
        "candidate": {},
        "planning": {},
    }
    road = {
        "xy": np.stack([np.linspace(0, 50, 64), np.zeros(64)], axis=-1).astype(np.float32),
        "heading": np.zeros(64, dtype=np.float32),
        "valid": np.ones(64, dtype=bool),
    }
    batch = build_online_batch(_toy_state(), 0, cfg, roadgraph=road)
    assert batch["cowp/candidates/valid"].shape == (1, 16)
    assert int(batch["cowp/candidates/valid"].sum()) >= 4
    assert int(batch["cowp/critical/valid"].sum()) >= 1
    assert int(batch["map/conflict_region_valid"].sum()) > 0
    assert "cowp/critical/input_index" in batch
