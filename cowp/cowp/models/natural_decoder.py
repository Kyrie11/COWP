from __future__ import annotations

import torch
from torch import nn


class NaturalDecoder(nn.Module):
    def __init__(self, d_model: int = 128, modes: int = 24, future_steps: int = 80):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, modes * future_steps * 7))
        self.logit = nn.Linear(d_model, modes)

    def forward(self, z_agent: torch.Tensor, critical_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        B, A = critical_indices.shape
        idx = critical_indices.clamp_min(0).long().unsqueeze(-1).expand(B, A, z_agent.shape[-1])
        z = torch.gather(z_agent, 1, idx)
        traj = self.head(z).reshape(B, A, self.modes, self.future_steps, 7)
        logits = self.logit(z)
        return {"traj": traj, "logits": logits}
