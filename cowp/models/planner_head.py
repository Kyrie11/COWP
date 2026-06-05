from __future__ import annotations

import torch
from torch import nn


class PlannerHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model + 4, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, z_candidate: torch.Tensor, ego_utility: torch.Tensor, witness_prob: torch.Tensor, opr: torch.Tensor, conventional_safe: torch.Tensor | None = None) -> torch.Tensor:
        # candidate-level aggregate over critical agents.
        max_wit = witness_prob.max(dim=-1).values if witness_prob.ndim == 3 else witness_prob
        min_opr = opr.min(dim=-1).values if opr.ndim == 3 else opr
        safe = torch.ones_like(max_wit) if conventional_safe is None else conventional_safe.float()
        aux = torch.stack([ego_utility.float(), max_wit.float(), min_opr.float(), safe], dim=-1)
        return self.score(torch.cat([z_candidate, aux], dim=-1)).squeeze(-1)
