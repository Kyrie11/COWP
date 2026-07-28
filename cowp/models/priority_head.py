from __future__ import annotations

import torch
from torch import nn


class PriorityClaimHead(nn.Module):
    """Candidate-agent priority-claim predictor.

    The output is a pairwise logit p(protected-priority claim | ego candidate, agent).
    It lets COWP learn which coercions should be hard-vetoed instead of treating
    every predicted witness as equally protected.
    """

    def __init__(self, d_model: int = 128, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        h = int(hidden or d_model)
        self.net = nn.Sequential(
            nn.Linear(4 * int(d_model) + 2, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

    def forward(
        self,
        z_agent: torch.Tensor,
        z_cand: torch.Tensor,
        critical_idx: torch.Tensor,
        witness_prob: torch.Tensor,
        opr: torch.Tensor,
    ) -> torch.Tensor:
        # z_agent [B,N,D], z_cand [B,K,D], critical_idx [B,A]
        B, K, D = z_cand.shape
        A = critical_idx.shape[1]
        safe_idx = critical_idx.clamp(0, max(z_agent.shape[1] - 1, 0)).long()
        gather_idx = safe_idx[..., None].expand(B, A, D)
        z_crit = torch.gather(z_agent, 1, gather_idx)  # [B,A,D]
        zc = z_cand[:, :, None, :].expand(B, K, A, D)
        za = z_crit[:, None, :, :].expand(B, K, A, D)
        pair = torch.cat(
            [
                zc,
                za,
                zc * za,
                torch.abs(zc - za),
                torch.nan_to_num(witness_prob.float(), nan=0.0, posinf=1.0, neginf=0.0).unsqueeze(-1),
                torch.nan_to_num(opr.float(), nan=1.0, posinf=1.0, neginf=0.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(pair).squeeze(-1)
