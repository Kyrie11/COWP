from __future__ import annotations

import torch
from torch import nn


class ResponseDecoder(nn.Module):
    def __init__(self, d_model: int = 128, responses: int = 32, future_steps: int = 80):
        super().__init__()
        self.responses = responses
        self.future_steps = future_steps
        self.pair = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.traj_head = nn.Linear(d_model, responses * future_steps * 7)
        self.safe_head = nn.Linear(d_model, responses)
        self.low_head = nn.Linear(d_model, responses)
        self.valid_head = nn.Linear(d_model, responses)
        self.mode_head = nn.Linear(d_model, responses)
        self.source_head = nn.Linear(d_model, responses * 4)
        self.burden_head = nn.Linear(d_model, responses * 7)  # total + 6 components

    def forward(
        self,
        z_agent: torch.Tensor,
        z_candidate: torch.Tensor,
        z_graph: torch.Tensor,
        critical_indices: torch.Tensor,
        *,
        decode_traj: bool = True,
    ) -> dict[str, torch.Tensor]:
        B, K, D = z_candidate.shape
        A = critical_indices.shape[1]
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, D)
        zcrit = torch.gather(z_agent, 1, idx)
        zc = z_candidate[:, :, None, :].expand(B, K, A, D)
        za = zcrit[:, None, :, :].expand(B, K, A, D)
        zg = z_graph[:, None, None, :].expand(B, K, A, D)
        pair = self.pair(torch.cat([zc, za, zg], dim=-1))
        burden = self.burden_head(pair).reshape(B, K, A, self.responses, 7)
        out = {
            "safe_logits": self.safe_head(pair),
            "low_logits": self.low_head(pair),
            "valid_logits": self.valid_head(pair),
            "mode_logits": self.mode_head(pair),
            "source_logits": self.source_head(pair).reshape(B, K, A, self.responses, 4),
            "burden_total": burden[..., 0].relu(),
            "burden_components": burden[..., 1:].relu().clamp(max=2.0),
        }
        if decode_traj:
            # This tensor is very large: [B,K,A,R,T,7].  Quick response training
            # can disable it when response_traj_l1 is set to zero, avoiding both
            # the giant linear projection and the matching label transfer.
            out["traj"] = self.traj_head(pair).reshape(B, K, A, self.responses, self.future_steps, 7)
        return out
