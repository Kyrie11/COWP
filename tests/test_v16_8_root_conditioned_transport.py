from __future__ import annotations

import numpy as np
import pytest
import torch

from cowp.label.safe_responses import _root_residual_trajectory
from cowp.models.losses import _root_low_safe_target, paper_aligned_supervision_batch


def _paper_batch(*, recovery: float) -> dict[str, torch.Tensor]:
    # One candidate, one protected agent, one natural root.  The root conflicts
    # with ego and is not directly retained, so OPR must equal q exactly.
    return {
        "cowp/candidates/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/candidates/conventional_safe": torch.ones(1, 1, dtype=torch.bool),
        "cowp/critical/valid": torch.ones(1, 1, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(1, 1, 1),
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "cowp/natural/beta": torch.full((1, 1), 0.65),
        "cowp/transport/mode_valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/mode_conflict": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/mode_retained_low_safe": torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        "cowp/transport/response_root_index": torch.zeros(1, 1, 1, 1, dtype=torch.long),
        "cowp/transport/root_low_safe_score": torch.full((1, 1, 1, 1), recovery),
        "cowp/response/valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_safe": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/is_low_burden": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "cowp/response/burden_total": torch.full((1, 1, 1, 1), 0.2),
        "cowp/witness/exists": torch.zeros(1, 1, 1, dtype=torch.bool),
    }


def test_paper_aligned_opr_transports_conflict_root_recovery():
    recovered = paper_aligned_supervision_batch(_paper_batch(recovery=1.0), {})
    lost = paper_aligned_supervision_batch(_paper_batch(recovery=0.0), {})
    assert recovered["cowp/witness/opr"].item() == pytest.approx(1.0)
    assert lost["cowp/witness/opr"].item() == pytest.approx(0.0)
    assert not recovered["cowp/witness/exists"].item()
    assert lost["cowp/witness/exists"].item()


def test_soft_root_target_is_used_without_collapsing_to_binary():
    batch = {"cowp/transport/root_low_safe_score": torch.tensor([[[[0.15, 0.80]]]])}
    target = _root_low_safe_target(batch, 2)
    assert target is not None
    assert torch.allclose(target, torch.tensor([[[[0.15, 0.80]]]]))


def test_root_residual_preserves_root_geometry_and_identity():
    t = np.arange(8, dtype=np.float32)
    root = np.zeros((8, 7), dtype=np.float32)
    root[:, 0] = t
    root[:, 2] = 0.0
    root[:, 3] = 10.0
    identity = _root_residual_trajectory(root, dt=0.1, accel=0.0, start_delay_s=0.0, duration_s=1.0)
    slowed = _root_residual_trajectory(root, dt=0.1, accel=-2.0, start_delay_s=0.0, duration_s=0.4)
    assert np.array_equal(identity, root)
    assert np.allclose(slowed[:, 1], root[:, 1])
    assert np.allclose(slowed[:, 2], root[:, 2])
    assert np.all(slowed[:, 3] <= root[:, 3] + 1.0e-6)


def test_planner_freezes_validated_transport_and_response_modules():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("cowp_train_script", Path("cowp/scripts/03_train.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = torch.nn.Linear(2, 2)
            self.candidate_encoder = torch.nn.Linear(2, 2)
            self.natural_decoder = torch.nn.Linear(2, 2)
            self.witness_decoder = torch.nn.Linear(2, 2)
            self.set_transport = torch.nn.Linear(2, 2)
            self.response_decoder = torch.nn.Linear(2, 2)

    model = Dummy()
    module._set_stage_freeze(
        model, "planner", False,
        freeze_transport_during_planner=True,
        freeze_response_during_planner=True,
    )
    assert not any(p.requires_grad for p in model.set_transport.parameters())
    assert not any(p.requires_grad for p in model.response_decoder.parameters())
    assert any(p.requires_grad for p in model.candidate_encoder.parameters())
