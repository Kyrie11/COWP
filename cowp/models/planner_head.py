from __future__ import annotations

import torch
from torch import nn


class PlannerHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model + 4, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(
        self,
        z_candidate: torch.Tensor,
        ego_utility: torch.Tensor,
        witness_prob: torch.Tensor,
        opr: torch.Tensor,
        conventional_safe: torch.Tensor | None = None,
        critical_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Candidate-level aggregate over critical agents.  Invalid padded critical
        # slots are not supervised by witness_loss; without this mask their random
        # logits can dominate max witness / min OPR and make learned selection fail.
        if witness_prob.ndim == 3:
            if critical_mask is not None:
                cm = critical_mask.bool()[:, None, :]
                witness_prob = torch.where(cm, witness_prob, torch.zeros_like(witness_prob))
                opr = torch.where(cm, opr, torch.ones_like(opr))
            max_wit = witness_prob.max(dim=-1).values
            min_opr = opr.min(dim=-1).values
        else:
            max_wit = witness_prob
            min_opr = opr
        safe = torch.ones_like(max_wit) if conventional_safe is None else conventional_safe.float()
        aux = torch.stack([ego_utility.float(), max_wit.float(), min_opr.float(), safe], dim=-1)
        return self.score(torch.cat([z_candidate, aux], dim=-1)).squeeze(-1)
