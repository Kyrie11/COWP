from __future__ import annotations

import numpy as np
import torch

from cowp.external_baselines.adapters import _candidate_to_ego_frame, _history_to_ego_frame
from cowp.scripts import __path__ as _scripts_path


def test_external_baseline_ego_frame_normalization() -> None:
    hist = torch.zeros(1, 1, 2, 11)
    hist[0, 0, :, 0] = torch.tensor([1001.0, 1002.0])
    hist[0, 0, :, 1] = torch.tensor([2000.0, 2000.0])
    hist[0, 0, :, 6] = torch.pi / 2
    hist[0, 0, :, 7] = 1.0
    hist[0, 0, :, 10] = 1.0
    origin = torch.tensor([[1002.0, 2000.0]])
    yaw0 = torch.tensor([torch.pi / 2])
    local = _history_to_ego_frame(hist, origin, yaw0)
    assert torch.allclose(local[0, 0, -1, :2], torch.zeros(2), atol=1e-5)
    assert abs(float(local[0, 0, -1, 6])) < 1e-5
    assert torch.allclose(local[0, 0, -1, 7:9], torch.tensor([0.0, -1.0]), atol=1e-5)

    cand = torch.zeros(1, 1, 2, 7)
    cand[0, 0, :, 0] = torch.tensor([1002.0, 1002.0])
    cand[0, 0, :, 1] = torch.tensor([2001.0, 2002.0])
    cand[0, 0, :, 2] = torch.pi / 2
    loc_cand = _candidate_to_ego_frame(cand, origin, yaw0)
    assert torch.allclose(loc_cand[0, 0, :, 0], torch.tensor([1.0, 2.0]), atol=1e-5)
    assert torch.allclose(loc_cand[0, 0, :, 1], torch.zeros(2), atol=1e-5)


def test_compact_table_loader_keeps_only_required_arrays(tmp_path) -> None:
    # Import the numeric script through importlib because its module name starts with a digit.
    import importlib
    mod = importlib.import_module("cowp.scripts.05_make_tables")
    p = tmp_path / "scene.npz"
    K, T = 3, 80
    np.savez(
        p,
        **{
            "cowp/candidates/trajectory": np.zeros((K, T, 7), dtype=np.float32),
            "cowp/candidates/valid": np.ones(K, dtype=bool),
            "cowp/candidates/conventional_safe": np.ones(K, dtype=bool),
            "cowp/candidates/ego_utility_prior": np.zeros(K, dtype=np.float32),
            "cowp/candidates/is_neutral": np.zeros(K, dtype=bool),
            "cowp/candidates/is_logged": np.zeros(K, dtype=bool),
            "cowp/candidates/macro_type": np.zeros(K, dtype=np.int32),
            "cowp/candidates/noncoercive_feasible": np.ones(K, dtype=bool),
            "cowp/candidates/false_safe": np.zeros(K, dtype=bool),
            "cowp/critical/valid": np.ones(1, dtype=bool),
            "cowp/natural/beta": np.zeros((1, 2), dtype=np.float32),
            "cowp/witness/exists": np.zeros((K, 1), dtype=bool),
            "cowp/witness/token": np.zeros((K, 1), dtype=np.int32),
            "cowp/witness/burden_total": np.zeros((K, 1), dtype=np.float32),
            "cowp/witness/min_safe_burden": np.zeros((K, 1), dtype=np.float32),
            "cowp/witness/opr": np.ones((K, 1), dtype=np.float32),
            "cowp/witness/c_i": np.zeros((K, 1), dtype=np.float32),
            "unused/massive_response_trajectory": np.zeros((64, 80, 7), dtype=np.float32),
        },
    )
    row = mod._read_compact_label(p)
    assert "unused/massive_response_trajectory" not in row
    assert row["cowp/candidates/trajectory"].shape == (K, 2, 7)
