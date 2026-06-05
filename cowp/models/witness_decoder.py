from __future__ import annotations

import torch
from torch import nn


class WitnessDecoder(nn.Module):
    def __init__(self, d_model: int = 128, token_count: int = 7):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.exist = nn.Linear(d_model, 1)
        self.token = nn.Linear(d_model, token_count)
        self.burden = nn.Linear(d_model, 7)
        self.interval = nn.Linear(d_model, 2)
        self.opr = nn.Linear(d_model, 1)
        self.ci = nn.Linear(d_model, 1)

    def forward(self, z_agent: torch.Tensor, z_candidate: torch.Tensor, z_graph: torch.Tensor, critical_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        B, K, D = z_candidate.shape
        A = critical_indices.shape[1]
        idx = critical_indices.clamp_min(0).long().unsqueeze(-1).expand(B, A, D)
        zcrit = torch.gather(z_agent, 1, idx)
        zc = z_candidate[:, :, None, :].expand(B, K, A, D)
        za = zcrit[:, None, :, :].expand(B, K, A, D)
        zg = z_graph[:, None, None, :].expand(B, K, A, D)
        h = self.pair(torch.cat([zc, za, zg], dim=-1))
        b = self.burden(h)
        return {
            "exist_logits": self.exist(h).squeeze(-1),
            "token_logits": self.token(h),
            "burden_total": b[..., 0].relu(),
            "burden_components": b[..., 1:].relu().clamp(max=2.0),
            "conflict_interval": self.interval(h).relu(),
            "opr": torch.sigmoid(self.opr(h).squeeze(-1)),
            "c_i": self.ci(h).squeeze(-1),
        }
