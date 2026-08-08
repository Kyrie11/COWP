from __future__ import annotations

import torch
from torch import nn


class WitnessDecoder(nn.Module):
    def __init__(self, d_model: int = 128, token_count: int = 7):
        super().__init__()
        # Cross-world witness representation: candidate-conditioned agent state,
        # candidate embedding, graph context, and root-scene natural latent.
        self.pair = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.exist = nn.Linear(d_model, 1)
        self.relevance = nn.Linear(d_model, 1)
        # Binary evidential head for uncertainty-aware coercion certification.
        # Channel 0 is non-witness evidence, channel 1 is witness evidence.
        self.evidence = nn.Linear(d_model, 2)
        self.token = nn.Linear(d_model, token_count)
        self.burden = nn.Linear(d_model, 7)
        self.interval = nn.Linear(d_model, 2)
        self.opr = nn.Linear(d_model, 1)
        self.ci = nn.Linear(d_model, 1)

    def forward(
        self,
        z_agent: torch.Tensor,
        z_candidate: torch.Tensor,
        z_graph: torch.Tensor,
        critical_indices: torch.Tensor,
        natural_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, K, D = z_candidate.shape
        A = critical_indices.shape[1]
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, D)
        zcrit = torch.gather(z_agent, 1, idx)
        zc = z_candidate[:, :, None, :].expand(B, K, A, D)
        za = zcrit[:, None, :, :].expand(B, K, A, D)
        zg = z_graph[:, None, None, :].expand(B, K, A, D)
        if natural_latent is None:
            zn = torch.zeros(B, A, D, device=z_agent.device, dtype=z_agent.dtype)
        else:
            zn = natural_latent
            if zn.shape[1] != A:
                raise ValueError(f"natural_latent has {zn.shape[1]} critical slots, expected {A}")
        zn = zn[:, None, :, :].expand(B, K, A, D)
        h = self.pair(torch.cat([zc, za, zg, zn], dim=-1))
        b = self.burden(h)
        evidence = torch.nn.functional.softplus(self.evidence(h)).clamp(max=50.0)
        beta = evidence[..., 0] + 1.0
        alpha = evidence[..., 1] + 1.0
        strength = alpha + beta
        raw_interval = self.interval(h).relu()
        start = torch.minimum(raw_interval[..., 0], raw_interval[..., 1])
        end = torch.maximum(raw_interval[..., 0], raw_interval[..., 1])
        return {
            "exist_logits": self.exist(h).squeeze(-1),
            "relevance_logit": self.relevance(h).squeeze(-1),
            "evidence_alpha": alpha,
            "evidence_beta": beta,
            "evidential_prob": alpha / strength.clamp_min(1e-6),
            "epistemic_uncertainty": 2.0 / strength.clamp_min(1e-6),
            "token_logits": self.token(h),
            "burden_total": b[..., 0].relu(),
            "burden_components": b[..., 1:].relu().clamp(max=2.0),
            "conflict_interval": torch.stack([start, end], dim=-1),
            "opr": torch.sigmoid(self.opr(h).squeeze(-1)),
            "c_i": self.ci(h).squeeze(-1),
        }
