from __future__ import annotations

"""Clean-room object-centric PlanT 2.0 adaptation for WOMD/COWP.

The official PlanT 2.0 model is CARLA-specific.  This implementation keeps its
central object-level Transformer planning abstraction while replacing CARLA
actor/route extraction by COWP's WOMD tensor adapter.  No source code is copied.
"""

from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


class COWPPlanT2(nn.Module):
    def __init__(self, future_len: int = 80, d_model: int = 128, num_heads: int = 8, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.future_len = int(future_len)
        self.d_model = int(d_model)
        self.agent_point = nn.Sequential(nn.Linear(9, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.object_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.route_point = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.map_point = nn.Sequential(nn.Linear(7, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.type_embedding = nn.Embedding(4, d_model)
        enc = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.temporal = nn.GRUCell(2 + d_model, d_model)
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.speed_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self.hazard_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
        w = valid.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        agents = inputs["agents"]
        valid = inputs.get("agent_valid", agents.abs().sum(dim=-1) > 0)
        ah = self.agent_point(agents)
        objs = self.object_head(self._masked_mean(ah, valid, -2)) + self.type_embedding.weight[0]
        route = inputs["route"]
        rv = route[..., 3] > 0.5
        rh = self.route_point(route) + self.type_embedding.weight[1]
        lanes = inputs["map_lanes"]
        lv = inputs.get("map_lanes_valid")
        if lv is None:
            lv = ~torch.eq(lanes, 0).all(dim=-1)
        lv = lv.bool()
        mh = self.map_point(lanes)
        mh = self._masked_mean(mh, lv, -2) + self.type_embedding.weight[2]
        tokens = torch.cat([objs, rh, mh], dim=1)
        pad = torch.cat([~valid.any(dim=-1), ~rv, ~lv.any(dim=-1)], dim=1)
        pad[:, 0] = False
        z = self.transformer(tokens, src_key_padding_mask=pad)
        return z, z[:, 0]

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, scene = self.encode(inputs)
        B = scene.shape[0]
        h = scene
        pos = torch.zeros(B, 2, device=scene.device, dtype=scene.dtype)
        traj = []
        speeds = []
        for _ in range(self.future_len):
            h = self.temporal(torch.cat([pos, scene], dim=-1), h)
            d = torch.tanh(self.delta_head(h))
            # PlanT is a waypoint planner; constrain forward motion softly but
            # allow stop/reverse-like corrections through a small signed term.
            dx = F.softplus(d[:, 0]) * 0.45
            dy = d[:, 1] * 0.30
            pos = pos + torch.stack([dx, dy], dim=-1)
            traj.append(pos)
            speeds.append(F.softplus(self.speed_head(h).squeeze(-1)))
        return {
            "trajectory": torch.stack(traj, dim=1),
            "speed": torch.stack(speeds, dim=1),
            "hazard_logit": self.hazard_head(scene).squeeze(-1),
            "scene_embedding": scene,
        }

    def predict_trajectory(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self(inputs)["trajectory"]

    def score_candidates(self, inputs: Mapping[str, torch.Tensor], candidates: torch.Tensor, candidate_valid: torch.Tensor) -> torch.Tensor:
        pred = self.predict_trajectory(inputs)
        T = min(candidates.shape[2], pred.shape[1])
        ade = torch.linalg.norm(candidates[:, :, :T, :2] - pred[:, None, :T], dim=-1).mean(dim=-1)
        return torch.where(candidate_valid, -ade, torch.full_like(ade, -1e9))


def plant2_loss(model: COWPPlanT2, inputs: Mapping[str, torch.Tensor], ego_future_xy: torch.Tensor, ego_future_valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(inputs)
    pred = out["trajectory"].float()
    gt = ego_future_xy[:, : pred.shape[1]].float()
    valid_b = ego_future_valid[:, : pred.shape[1]].bool()
    valid = valid_b.float()
    sample_valid = valid_b.any(dim=-1)
    if not bool(sample_valid.any()):
        zero = pred.sum() * 0.0 + out["speed"].sum() * 0.0 + out["hazard_logit"].sum() * 0.0
        return zero, {"plannerADE": float("nan"), "plannerFDE": float("nan"), "valid_samples": 0.0}
    gt_safe = torch.where(valid_b[..., None], gt, pred.detach())
    reg = F.smooth_l1_loss(pred, gt_safe, reduction="none").sum(-1)
    reg = (reg * valid).sum() / valid.sum().clamp_min(1.0)
    # Speed supervision requires two consecutive valid logged positions.  Using
    # only the destination validity can turn a missing predecessor into a huge
    # artificial speed target.
    gt_speed = torch.zeros_like(valid)
    speed_valid = torch.zeros_like(valid_b)
    if gt.shape[1] > 1:
        transition_valid = valid_b[:, 1:] & valid_b[:, :-1]
        delta = torch.where(transition_valid[..., None], gt[:, 1:] - gt[:, :-1], torch.zeros_like(gt[:, 1:]))
        gt_speed[:, 1:] = torch.linalg.norm(delta, dim=-1) / 0.1
        speed_valid[:, 1:] = transition_valid
        gt_speed[:, 0] = gt_speed[:, 1]
        speed_valid[:, 0] = transition_valid[:, 0]
    sp_per = F.smooth_l1_loss(out["speed"].float(), gt_speed, reduction="none")
    sp = (sp_per * speed_valid.float()).sum() / speed_valid.float().sum().clamp_min(1.0)
    # Collision-risk auxiliary target: whether logged ego comes within a compact
    # radius of any logged neighboring future.  It is observational, not COWP.
    neigh = inputs.get("neighbors_future_xy")
    neigh_valid = inputs.get("neighbors_future_valid")
    hazard_loss = pred.sum() * 0.0
    if neigh is not None and neigh_valid is not None and neigh.numel() > 0:
        T = min(gt.shape[1], neigh.shape[2])
        pair_valid = neigh_valid[:, :, :T].bool() & valid_b[:, None, :T]
        pair_delta = torch.where(
            pair_valid[..., None], gt[:, None, :T] - neigh[:, :, :T], torch.zeros_like(neigh[:, :, :T])
        )
        d = torch.linalg.norm(pair_delta, dim=-1)
        d = torch.where(pair_valid, d, torch.full_like(d, 1e6))
        target = (d.amin(dim=-1).amin(dim=-1) < 3.0).float()
        hazard_loss = F.binary_cross_entropy_with_logits(out["hazard_logit"].float()[sample_valid], target[sample_valid])
    loss = reg + 0.05 * sp + 0.05 * hazard_loss
    metric_target = torch.where(valid_b[..., None], gt, pred.detach())
    d = torch.linalg.norm(pred - metric_target, dim=-1)
    d = torch.where(valid_b, d, torch.zeros_like(d))
    ade = d.sum() / valid.sum().clamp_min(1.0)
    rows = torch.arange(gt.shape[0], device=gt.device)
    last = valid.sum(dim=-1).long().clamp_min(1) - 1
    fde = d[rows[sample_valid], last[sample_valid]].mean() if bool(sample_valid.any()) else d.sum() * 0.0
    return loss, {
        "plannerADE": float(ade.detach().cpu()),
        "plannerFDE": float(fde.detach().cpu()),
        "ego_reg": float(reg.detach().cpu()),
        "speed_reg": float(sp.detach().cpu()),
        "hazard_bce": float(hazard_loss.detach().cpu()),
        "valid_samples": float(sample_valid.float().sum().detach().cpu()),
    }
