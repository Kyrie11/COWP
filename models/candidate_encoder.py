from __future__ import annotations

import torch
from torch import nn


class CandidateEncoder(nn.Module):
    def __init__(self, d_model: int = 128, macro_count: int = 13, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.GRU(input_size=7, hidden_size=d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.macro_embed = nn.Embedding(macro_count, d_model)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, traj: torch.Tensor, macro_type: torch.Tensor) -> torch.Tensor:
        B, K, T, D = traj.shape
        h, _ = self.temporal(traj.reshape(B * K, T, D))
        pooled = h.mean(dim=1).reshape(B, K, -1)
        macro_idx = macro_type.long().clamp(0, self.macro_embed.num_embeddings - 1)
        macro = self.macro_embed(macro_idx)
        return self.proj(pooled + macro)
