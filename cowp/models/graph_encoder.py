from __future__ import annotations

import torch
from torch import nn


class GraphEncoder(nn.Module):
    """Heterogeneous-style encoder over ego/agent/lane/conflict features.

    The module accepts padded tensors and masks. It uses type embeddings and a transformer
    encoder; relation-specific details are carried by caller-provided features and masks.
    """

    def __init__(self, d_state: int = 11, d_model: int = 128, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.agent_proj = nn.Linear(d_state, d_model)
        self.candidate_proj = nn.Linear(7, d_model)
        self.conflict_proj = nn.Linear(8, d_model)
        self.type_embed = nn.Embedding(4, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, agent_history: torch.Tensor, agent_mask: torch.Tensor, candidate_traj: torch.Tensor | None = None, candidate_mask: torch.Tensor | None = None, conflict_regions: torch.Tensor | None = None, conflict_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # agent_history: [B,N,Th,D]. Use a validity-weighted temporal mean when
        # the last channel carries WOMD validity. Averaging invalid zero-filled
        # rows was silently biasing agent embeddings toward the origin.
        if agent_history.ndim == 4:
            if agent_history.shape[-1] >= 11:
                hist_valid = agent_history[..., 10:11].clamp(0.0, 1.0)
                denom = hist_valid.sum(dim=2).clamp_min(1.0)
                x_agent = (agent_history * hist_valid).sum(dim=2) / denom
                empty = hist_valid.sum(dim=2).squeeze(-1) <= 0
                if empty.any():
                    x_agent = torch.where(empty.unsqueeze(-1), agent_history.mean(dim=2), x_agent)
            else:
                x_agent = agent_history.mean(dim=2)
        else:
            x_agent = agent_history
        z_agent = self.agent_proj(x_agent) + self.type_embed(torch.zeros_like(agent_mask.long()))
        pieces = [z_agent]
        masks = [agent_mask.bool()]
        if candidate_traj is not None:
            z_cand = self.candidate_proj(candidate_traj.mean(dim=2)) + self.type_embed(torch.ones(candidate_traj.shape[:2], device=candidate_traj.device, dtype=torch.long))
            pieces.append(z_cand)
            masks.append(candidate_mask.bool() if candidate_mask is not None else torch.ones(candidate_traj.shape[:2], device=candidate_traj.device, dtype=torch.bool))
        else:
            z_cand = None
        if conflict_regions is not None:
            z_conf = self.conflict_proj(conflict_regions) + self.type_embed(torch.full(conflict_regions.shape[:2], 2, device=conflict_regions.device, dtype=torch.long))
            pieces.append(z_conf)
            masks.append(conflict_mask.bool() if conflict_mask is not None else torch.ones(conflict_regions.shape[:2], device=conflict_regions.device, dtype=torch.bool))
        else:
            z_conf = None
        z = torch.cat(pieces, dim=1)
        mask = torch.cat(masks, dim=1)
        enc = self.encoder(z, src_key_padding_mask=~mask)
        enc = self.norm(enc)
        n_agent = z_agent.shape[1]
        out = {"z_all": enc, "mask_all": mask, "z_agent": enc[:, :n_agent]}
        offset = n_agent
        if z_cand is not None:
            out["z_candidate_context"] = enc[:, offset : offset + z_cand.shape[1]]
            offset += z_cand.shape[1]
        if z_conf is not None:
            out["z_conflict"] = enc[:, offset : offset + z_conf.shape[1]]
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        out["z_graph"] = (enc * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        out["z_ego"] = enc[:, 0]
        return out
