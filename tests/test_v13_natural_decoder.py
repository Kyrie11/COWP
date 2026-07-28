from __future__ import annotations

import torch

from cowp.models.natural_decoder import NaturalDecoder
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, policy_diagnostic_summary


def test_temporal_kinematic_decoder_starts_from_constant_velocity_baseline():
    decoder = NaturalDecoder(d_model=16, modes=3, future_steps=4, decoder_type="temporal_kinematic")
    z_agent = torch.zeros(1, 2, 16)
    critical_indices = torch.tensor([[1]])
    anchor7 = torch.tensor([[[10.0, 20.0, 0.3, 2.0, -1.0, 4.5, 1.9]]])

    out = decoder(z_agent, critical_indices, anchor7=anchor7, dt=0.5)
    traj_offset = out["traj"]

    expected_x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_y = torch.tensor([-0.5, -1.0, -1.5, -2.0])
    assert torch.allclose(traj_offset[0, 0, :, :, 0], expected_x.expand(3, -1))
    assert torch.allclose(traj_offset[0, 0, :, :, 1], expected_y.expand(3, -1))
    assert torch.count_nonzero(traj_offset[..., 2:]) == 0


def test_policy_diagnostics_mark_mechanism_values_as_proxy_only():
    out = policy_diagnostic_summary(
        [{"policy_diagnostics": [{"max_witness_prob": 0.7, "witness_threshold": 0.5}]}]
    )
    assert out["ClosedLoopMechanismProxyOnly"] == 1.0
    assert out["ClosedLoopMechanismGroundTruthAvailable"] == 0.0


def test_label_metrics_missing_optional_witness_arrays_do_not_crash():
    label = {
        "cowp/candidates/trajectory": torch.zeros(2, 4, 7).numpy(),
        "cowp/candidates/valid": torch.tensor([True, True]).numpy(),
        "cowp/candidates/conventional_safe": torch.tensor([True, True]).numpy(),
        "cowp/critical/valid": torch.tensor([True]).numpy(),
        "cowp/witness/exists": torch.tensor([[False], [False]]).numpy(),
    }
    out = metrics_from_labels([0], [label])
    assert out["CR"] == 0.0
    assert out["CBS"] == 0.0
    assert out["OPR"] == 1.0
