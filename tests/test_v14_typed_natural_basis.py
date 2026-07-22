from __future__ import annotations

import torch

from cowp.core.constants import NaturalSource
from cowp.models.losses import natural_loss
from cowp.models.natural_decoder import NaturalDecoder


def test_typed_basis_has_stable_balanced_source_identity_and_noncollapsed_geometry():
    decoder = NaturalDecoder(
        d_model=16, modes=24, future_steps=20,
        decoder_type="typed_kinematic_residual",
    )
    z_agent = torch.zeros(1, 2, 16)
    critical_indices = torch.tensor([[1]])
    anchor = torch.tensor([[[10.0, 20.0, 0.0, 8.0, 0.0, 4.5, 1.9]]])
    out = decoder(z_agent, critical_indices, anchor7=anchor, dt=0.1)

    assert out["traj"].shape == (1, 1, 24, 20, 7)
    assert torch.equal(
        torch.bincount(out["mode_source"], minlength=4),
        torch.tensor([8, 8, 8, 0]),
    )
    # Zero-initialized residual means the first forward is exactly the analytic bank.
    assert torch.count_nonzero(out["residual"]) == 0
    assert torch.allclose(out["traj"], out["base_traj"])
    # The old v13 decoder initialized every root to the same curve; v14 must not.
    endpoints = out["traj"][0, 0, :, -1, :2]
    assert torch.unique(torch.round(endpoints * 1000), dim=0).shape[0] >= 12


def test_typed_source_bias_assigns_each_mode_to_its_declared_source():
    decoder = NaturalDecoder(d_model=8, modes=24, future_steps=4, decoder_type="typed_kinematic_residual")
    out = decoder(
        torch.zeros(1, 1, 8), torch.tensor([[0]]),
        anchor7=torch.tensor([[[0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 2.0]]]),
        dt=0.1,
    )
    predicted_source = out["source_logits"].argmax(dim=-1)[0, 0]
    assert torch.equal(predicted_source, out["mode_source"])


def test_natural_loss_restricts_matching_to_same_source():
    # Geometrically, mode 0 is close to the NEU GT and mode 1 is far away.
    # Typed matching must nevertheless use the declared NEU mode 1.
    pred_traj = torch.zeros(1, 1, 2, 2, 7)
    pred_traj[0, 0, 0, :, 0] = 0.0  # OBS
    pred_traj[0, 0, 1, :, 0] = 10.0  # NEU
    gt = torch.zeros(1, 1, 1, 2, 7)
    batch = {
        "cowp/natural/traj": gt,
        "cowp/natural/valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "cowp/natural/weight": torch.ones(1, 1, 1),
        "cowp/natural/source": torch.full((1, 1, 1), int(NaturalSource.NEU), dtype=torch.long),
        "cowp/natural/priority_preserved": torch.zeros(1, 1, 1),
        "cowp/critical/valid": torch.ones(1, 1, dtype=torch.bool),
    }
    source_logits = torch.full((1, 1, 2, 4), -3.0)
    source_logits[0, 0, 0, int(NaturalSource.OBS)] = 3.0
    source_logits[0, 0, 1, int(NaturalSource.NEU)] = 3.0
    pred = {
        "traj": pred_traj,
        "base_traj": pred_traj.clone(),
        "residual": torch.zeros_like(pred_traj),
        "logits": torch.zeros(1, 1, 2),
        "source_logits": source_logits,
        "priority_logits": torch.zeros(1, 1, 2),
        "mode_source": torch.tensor([int(NaturalSource.OBS), int(NaturalSource.NEU)]),
    }
    losses = natural_loss(pred, batch, {"natural_dt": 0.1})
    assert torch.isclose(losses["untyped_traj"], torch.tensor(0.0))
    assert torch.isclose(losses["traj"], torch.tensor(10.0))
