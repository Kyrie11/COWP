from __future__ import annotations

"""Clean-room WOMD/COWP adaptation of PLUTO's planning ideas.

This module intentionally contains no copied PLUTO source.  It preserves the
paper-level design that is relevant to an object/vector WOMD interface:
  * vectorized scene encoding,
  * factorized longitudinal/lateral planning queries,
  * multi-modal imitation planning,
  * auxiliary imitation supervision, and
  * an optional contrastive scene-consistency loss.

The original PLUTO implementation targets nuPlan.  The surrounding COWP
adapter supplies ego-centric WOMD agent/map/route tensors instead.
"""

from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


def _valid_mean(x: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    w = valid.to(x.dtype)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


class PolylineEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        h = self.net(x)
        if valid is None:
            pooled = h.mean(dim=-2)
        else:
            pooled = _valid_mean(h, valid, dim=-2)
        return self.norm(pooled)


class COWPPLUTO(nn.Module):
    """PLUTO-style factorized query planner for WOMD object/vector inputs."""

    def __init__(
        self,
        future_len: int = 80,
        d_model: int = 128,
        num_heads: int = 8,
        encoder_layers: int = 4,
        lateral_queries: int = 4,
        longitudinal_queries: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.d_model = int(d_model)
        self.lateral_queries = int(lateral_queries)
        self.longitudinal_queries = int(longitudinal_queries)
        self.num_modes = self.lateral_queries * self.longitudinal_queries

        self.agent_encoder = PolylineEncoder(9, d_model)
        self.map_encoder = PolylineEncoder(7, d_model)
        self.route_encoder = PolylineEncoder(4, d_model)
        self.token_type = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.scene_encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.lat_query = nn.Parameter(torch.randn(self.lateral_queries, d_model) * 0.02)
        self.lon_query = nn.Parameter(torch.randn(self.longitudinal_queries, d_model) * 0.02)
        self.query_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mode_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.traj_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, self.future_len * 2),
        )
        self.aux_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.future_len * 2))
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 64))

    def encode_scene(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        agents = inputs["agents"]  # [B,N,T,9]
        agent_valid = inputs.get("agent_valid")
        if agent_valid is None:
            agent_valid = agents.abs().sum(dim=-1) > 0
        amap = inputs["map_lanes"]  # [B,L,P,7]
        map_valid = amap[..., :2].abs().sum(dim=-1) > 0
        route = inputs["route"]  # [B,R,4]
        route_valid = route[..., 3] > 0.5

        a = self.agent_encoder(agents, agent_valid) + self.token_type.weight[0]
        m = self.map_encoder(amap, map_valid) + self.token_type.weight[1]
        r = self.route_encoder(route[:, None], route_valid[:, None]) + self.token_type.weight[2]
        tokens = torch.cat([a, m, r], dim=1)
        pad = torch.cat([
            ~agent_valid.any(dim=-1),
            ~map_valid.any(dim=-1),
            ~route_valid.any(dim=-1, keepdim=True),
        ], dim=1)
        # TransformerEncoder can produce NaNs if a row is fully masked; ego is
        # always expected valid, but make the contract fail-soft for smoke tests.
        pad[:, 0] = False
        enc = self.scene_encoder(tokens, src_key_padding_mask=pad)
        scene = enc[:, 0]
        return enc, scene

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enc, scene = self.encode_scene(inputs)
        B = enc.shape[0]
        lat = self.lat_query[None, :, None, :].expand(B, -1, self.longitudinal_queries, -1)
        lon = self.lon_query[None, None, :, :].expand(B, self.lateral_queries, -1, -1)
        q = (lat + lon + scene[:, None, None, :]).reshape(B, self.num_modes, self.d_model)
        q2, _ = self.query_cross(q, enc, enc, need_weights=False)
        q = q + q2
        # Decode local increments and integrate them.  This gives a stable
        # trajectory parameterization while allowing lateral/longitudinal modes.
        delta = self.traj_head(q).reshape(B, self.num_modes, self.future_len, 2)
        delta = torch.tanh(delta)
        delta_x = F.softplus(delta[..., 0]) * 0.45
        delta_y = delta[..., 1] * 0.30
        traj = torch.cumsum(torch.stack([delta_x, delta_y], dim=-1), dim=-2)
        scores = self.mode_head(q).squeeze(-1)
        aux = self.aux_head(scene).reshape(B, self.future_len, 2)
        aux = torch.cumsum(torch.stack([F.softplus(torch.tanh(aux[..., 0])) * 0.45, torch.tanh(aux[..., 1]) * 0.30], dim=-1), dim=-2)
        return {"trajectories": traj, "scores": scores, "aux_trajectory": aux, "scene_embedding": scene}

    def predict_trajectory(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        out = self(inputs)
        idx = out["scores"].argmax(dim=-1)
        return out["trajectories"][torch.arange(idx.shape[0], device=idx.device), idx]

    def score_candidates(self, inputs: Mapping[str, torch.Tensor], candidates: torch.Tensor, candidate_valid: torch.Tensor) -> torch.Tensor:
        out = self(inputs)
        modes = out["trajectories"]
        probs = out["scores"].log_softmax(dim=-1)
        T = min(candidates.shape[2], modes.shape[2])
        dist = torch.linalg.norm(candidates[:, :, None, :T, :2] - modes[:, None, :, :T], dim=-1).mean(dim=-1)
        score = (probs[:, None] - dist).amax(dim=-1)
        return torch.where(candidate_valid, score, torch.full_like(score, -1e9))


def _best_mode(traj: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid[:, None].float()
    dist = torch.linalg.norm(traj - gt[:, None], dim=-1) * w
    ade = dist.sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)
    return ade.argmin(dim=-1)


def pluto_loss(
    model: COWPPLUTO,
    inputs: Mapping[str, torch.Tensor],
    ego_future_xy: torch.Tensor,
    ego_future_valid: torch.Tensor,
    *,
    contrast_weight: float = 0.05,
    aux_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(inputs)
    traj = out["trajectories"].float()
    scores = out["scores"].float()
    gt = ego_future_xy[:, : traj.shape[2]].float()
    valid = ego_future_valid[:, : traj.shape[2]].bool()
    best = _best_mode(traj, gt, valid)
    rows = torch.arange(gt.shape[0], device=gt.device)
    pred = traj[rows, best]
    v = valid.float()
    reg = F.smooth_l1_loss(pred, gt, reduction="none").sum(-1)
    reg = (reg * v).sum() / v.sum().clamp_min(1.0)
    cls = F.cross_entropy(scores, best, label_smoothing=0.1)
    aux = out["aux_trajectory"].float()[:, : gt.shape[1]]
    aux_l = F.smooth_l1_loss(aux, gt, reduction="none").sum(-1)
    aux_l = (aux_l * v).sum() / v.sum().clamp_min(1.0)

    contrast = scores.sum() * 0.0
    if contrast_weight > 0 and gt.shape[0] > 1:
        # PLUTO's CIL motivation is scene consistency under behavior-preserving
        # perturbations.  We use an agent-dropout view that needs no COWP labels.
        aug = dict(inputs)
        agents = inputs["agents"].clone()
        if agents.shape[1] > 1:
            keep = (torch.rand(agents.shape[:2], device=agents.device) > 0.15)
            keep[:, 0] = True
            agents = agents * keep[:, :, None, None]
            if "agent_valid" in inputs:
                # A dropped actor must be dropped from the transformer padding
                # mask as well.  Previously only its features were zeroed, so
                # MLP biases still created an apparently valid actor token.
                aug["agent_valid"] = inputs["agent_valid"] & keep[:, :, None]
        aug["agents"] = agents
        _, z2 = model.encode_scene(aug)
        z1 = F.normalize(model.proj(out["scene_embedding"].float()), dim=-1)
        z2 = F.normalize(model.proj(z2.float()), dim=-1)
        logits = z1 @ z2.T / 0.1
        target = torch.arange(logits.shape[0], device=logits.device)
        contrast = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    loss = reg + cls + aux_weight * aux_l + contrast_weight * contrast
    pred_best = traj[rows, scores.argmax(dim=-1)]
    d = torch.linalg.norm(pred_best - gt, dim=-1) * v
    ade = d.sum() / v.sum().clamp_min(1.0)
    last = v.sum(dim=-1).long().clamp_min(1) - 1
    sample_valid = valid.any(dim=-1)
    fde = d[rows[sample_valid], last[sample_valid]].mean() if bool(sample_valid.any()) else d.sum() * 0.0
    return loss, {
        "plannerADE": float(ade.detach().cpu()),
        "plannerFDE": float(fde.detach().cpu()),
        "mode_ce": float(cls.detach().cpu()),
        "ego_reg": float(reg.detach().cpu()),
        "aux_reg": float(aux_l.detach().cpu()),
        "contrast": float(contrast.detach().cpu()),
    }
