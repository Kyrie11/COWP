from __future__ import annotations

import torch
from torch import nn


class NaturalDecoder(nn.Module):
    """Multi-branch natural alternative decoder.

    Besides mode trajectories and mixture logits, the decoder predicts a source
    branch (observed / ego-neutral / priority-preserving) and whether the branch
    preserves local priority.  These heads make the paper's natural-branch loss
    terms directly trainable instead of treating all alternatives as an unordered
    trajectory bank.
    """

    def __init__(self, d_model: int = 128, modes: int = 24, future_steps: int = 80, source_count: int = 4):
        super().__init__()
        self.modes = modes
        self.future_steps = future_steps
        self.source_count = source_count
        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, modes * future_steps * 7)
        self.logit = nn.Linear(d_model, modes)
        self.source_logit = nn.Linear(d_model, modes * source_count)
        self.priority_logit = nn.Linear(d_model, modes)

    def forward(self, z_agent: torch.Tensor, critical_indices: torch.Tensor, *, decode_traj: bool = True) -> dict[str, torch.Tensor]:
        B, A = critical_indices.shape
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, z_agent.shape[-1])
        z = torch.gather(z_agent, 1, idx)
        h = self.shared(z)
        logits = self.logit(h)
        source_logits = self.source_logit(h).reshape(B, A, self.modes, self.source_count)
        priority_logits = self.priority_logit(h)
        out = {"latent": h, "logits": logits, "source_logits": source_logits, "priority_logits": priority_logits}
        if decode_traj:
            out["traj"] = self.head(h).reshape(B, A, self.modes, self.future_steps, 7)
        return out
