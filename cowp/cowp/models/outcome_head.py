from __future__ import annotations

import torch
from torch import nn


class OutcomeRiskHead(nn.Module):
    """Closed-loop candidate outcome predictor.

    Outputs collision/offroad logits and a non-negative log-divergence proxy. It is
    trained only when attached Waymax replay labels exist, and is otherwise ignored
    by default at inference.
    """

    def __init__(self, d_model: int = 128, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        h = int(hidden or d_model)
        self.net = nn.Sequential(
            nn.Linear(int(d_model), h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 3),
        )

    def forward(self, z_cand: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(z_cand)
        return {
            "collision_logit": raw[..., 0],
            "offroad_logit": raw[..., 1],
            "logdiv": torch.nn.functional.softplus(raw[..., 2]),
        }
