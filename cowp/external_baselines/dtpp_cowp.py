from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.permute(1, 0, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[-2]])


class AgentEncoder(nn.Module):
    def __init__(self, agent_dim: int, dim: int = 256):
        super().__init__()
        self.motion = nn.LSTM(agent_dim, dim, 2, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        traj, _ = self.motion(inputs)
        return traj[:, -1]


class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim: int, map_len: int, dim: int = 256):
        super().__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, dim))
        self.position_encode = PositionalEncoding(dim, max_len=map_len)

    def segment_map(
        self, map_tensor: torch.Tensor, map_encoding: torch.Tensor,
        point_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N_e, N_p, D = map_encoding.shape
        points_per_segment = 10
        if N_p % points_per_segment != 0:
            raise ValueError(f"DTPP map points must be divisible by {points_per_segment}, got {N_p}")
        if point_valid is None:
            point_valid = ~torch.eq(map_tensor, 0).all(dim=-1)
        valid = point_valid.bool().reshape(B, N_e, N_p // points_per_segment, points_per_segment)
        enc = map_encoding.reshape(B, N_e, N_p // points_per_segment, points_per_segment, D)
        floor = torch.finfo(map_encoding.dtype).min
        enc = enc.masked_fill(~valid[..., None], floor).max(dim=3).values
        segment_valid = valid.any(dim=3)
        enc = torch.where(segment_valid[..., None], enc, torch.zeros_like(enc))
        return enc.reshape(B, -1, D), (~segment_valid).reshape(B, -1)

    def forward(self, x: torch.Tensor, point_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.segment_map(x, self.position_encode(self.point_net(x)), point_valid)


class CrossAttention(nn.Module):
    def __init__(self, heads: int = 8, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Match the public DTPP CrossAttention block: the attention output is
        # normalized before the FFN; there is no extra query residual here.
        out, _ = self.cross_attention(query, key, value, attn_mask=mask)
        out = self.norm_1(out)
        return self.norm_2(out + self.dropout(self.ffn(out)))


class AgentDecoder(nn.Module):
    def __init__(self, max_time: int, max_branch: int, dim: int):
        super().__init__()
        self.max_time = int(max_time)
        self.max_branch = int(max_branch)
        self.traj_decoder = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, 3 * 10))

    def forward(self, encoding: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        encoding = torch.reshape(encoding, (encoding.shape[0], self.max_branch, self.max_time, 512))
        agent_traj = self.traj_decoder(encoding).reshape(encoding.shape[0], self.max_branch, self.max_time * 10, 3)
        agent_traj += current_state[:, None, None, :3]
        return agent_traj


class ScoreDecoder(nn.Module):
    def __init__(self, variable_cost: bool = False):
        super().__init__()
        self.n_latent_features = 4
        self.variable_cost = bool(variable_cost)
        self.interaction_feature_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self.n_latent_features), nn.Sigmoid())
        self.weights_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self.n_latent_features + 4), nn.Softplus())

    def get_hardcoded_features(self, ego_traj: torch.Tensor, max_time: int) -> torch.Tensor:
        speed = ego_traj[:, :, :max_time, 3]
        acceleration = ego_traj[:, :, :max_time, 4]
        jerk = torch.diff(acceleration, dim=-1) / 0.1
        jerk = torch.cat((jerk[:, :, :1], jerk), dim=-1)
        curvature = ego_traj[:, :, :max_time, 5]
        lateral_acc = speed ** 2 * curvature
        speed = -speed.mean(-1).clip(0, 15) / 15
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6
        lateral_acc = lateral_acc.abs().mean(-1).clip(0, 5) / 5
        return torch.stack((speed, acceleration, jerk, lateral_acc), dim=-1)

    def calculate_collision(self, ego_traj: torch.Tensor, agent_traj: torch.Tensor, agents_states: torch.Tensor, max_time: int) -> torch.Tensor:
        # WOMD adapter appends an explicit validity channel at index 10.  A
        # stopped neighbor can otherwise be all-zero in ego-centric kinematics
        # and must still participate in interaction/collision scoring.
        agent_mask = agents_states[..., 10] > 0.5
        dist = torch.linalg.norm(ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], dim=-1)
        return (torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None]).sum(-1).sum(-1)

    def get_latent_interaction_features(self, ego_traj: torch.Tensor, agent_traj: torch.Tensor, agents_states: torch.Tensor, max_time: int) -> torch.Tensor:
        agent_mask = agents_states[..., 10] > 0.5
        ego_yaw = ego_traj[:, None, :max_time, 2]
        relative_yaw = torch.atan2(torch.sin(agent_traj[:, :, :max_time, 2] - ego_yaw), torch.cos(agent_traj[:, :, :max_time, 2] - ego_yaw))
        rel = agent_traj[:, :, :max_time, :2] - ego_traj[:, None, :max_time, :2]
        cos_y = torch.cos(ego_yaw)
        sin_y = torch.sin(ego_yaw)
        relative_pos = torch.stack([rel[..., 0] * cos_y + rel[..., 1] * sin_y, -rel[..., 0] * sin_y + rel[..., 1] * cos_y], dim=-1)
        agent_velocity = torch.diff(agent_traj[:, :, :max_time, :2], dim=-2) / 0.1
        agent_velocity = torch.cat((agent_velocity[:, :, :1, :], agent_velocity), dim=-2)
        ego_vx = ego_traj[:, :max_time, 3] * torch.cos(ego_traj[:, :max_time, 2])
        ego_vy = ego_traj[:, :max_time, 3] * torch.sin(ego_traj[:, :max_time, 2])
        dvx = agent_velocity[..., 0] - ego_vx[:, None]
        dvy = agent_velocity[..., 1] - ego_vy[:, None]
        relative_velocity = torch.stack([dvx * cos_y + dvy * sin_y, -dvx * sin_y + dvy * cos_y], dim=-1)
        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1)
        agent_attributes = agents_states[:, :, None, 6:].expand(-1, -1, relative_attributes.shape[2], -1)
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1) * agent_mask[:, :, None, None]
        features = self.interaction_feature_encoder(attributes)
        features = features.max(1).values.mean(1)
        return self.interaction_feature_decoder(features)

    def forward(self, ego_traj: torch.Tensor, ego_encoding: torch.Tensor, agents_traj: torch.Tensor, agents_states: torch.Tensor, timesteps: int, candidate_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized branch scoring.

        The public DTPP formulation scores each tree branch independently with
        the same learned/hard-coded feature functions.  The earlier adapter used
        a Python loop over K branches; flattening BxK performs the exact same
        tensor operations in one GPU launch family and removes a major K=30
        inference/training bottleneck.
        """
        ego_traj_features = self.get_hardcoded_features(ego_traj, timesteps)
        if not self.variable_cost:
            ego_encoding = torch.ones_like(ego_encoding)
        weights = self.weights_decoder(ego_encoding)
        # Prefer the explicit COWP proposal-valid mask.  It is the only reliable
        # way to distinguish an intentionally stationary stop trajectory (which
        # can be all zeros in the ego frame) from a padded branch.
        if candidate_valid is None:
            ego_mask = ego_traj.abs().sum(dim=-1).sum(dim=-1) > 0
        else:
            ego_mask = candidate_valid.bool()
        B, K = ego_traj.shape[:2]
        T = min(int(timesteps), int(ego_traj.shape[2]), int(agents_traj.shape[3]))
        ego_flat = ego_traj[:, :, :T].reshape(B * K, T, ego_traj.shape[-1])
        agents_flat = agents_traj[:, :, :, :T].reshape(B * K, agents_traj.shape[2], T, agents_traj.shape[-1])
        states_flat = agents_states[:, None].expand(B, K, *agents_states.shape[1:]).reshape(B * K, *agents_states.shape[1:])
        latent = self.get_latent_interaction_features(ego_flat, agents_flat, states_flat, T).reshape(B, K, -1)
        collision = self.calculate_collision(ego_flat, agents_flat, states_flat, T).reshape(B, K)
        feat = torch.cat((ego_traj_features, latent), dim=-1)
        scores = -torch.sum(feat * weights[:, None, :], dim=-1) - 10.0 * collision
        scores = torch.where(ego_mask, scores, torch.full_like(scores, -1e9))
        return scores, weights


class DTPPEncoder(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.agent_encoder = AgentEncoder(11, dim)
        self.ego_encoder = AgentEncoder(7, dim)
        self.lane_encoder = VectorMapEncoder(7, 50, dim)
        self.crosswalk_encoder = VectorMapEncoder(3, 30, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ego = inputs["ego_agent_past"]
        neighbors = inputs["neighbor_agents_past"]
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        # Inputs are in the current SDC frame.  A stopped ego therefore has
        # x=y=yaw=vx=vy=0 by construction and must not be mistaken for padding.
        # Use the explicit validity channels emitted by adapters.py instead.
        ego_valid = ego[:, -1, 6] > 0.5
        neighbor_valid = neighbors[:, :, -1, 10] > 0.5
        actors_mask = torch.cat([~ego_valid[:, None], ~neighbor_valid], dim=1)
        lanes, lanes_mask = self.lane_encoder(inputs["map_lanes"], inputs.get("map_lanes_valid"))
        cross, cross_mask = self.crosswalk_encoder(inputs["map_crosswalks"], inputs.get("map_crosswalks_valid"))
        inp = torch.cat([encoded_actors, lanes, cross], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, cross_mask], dim=1)
        return {"encoding": self.fusion_encoder(inp, src_key_padding_mask=mask), "mask": mask}


class DTPPDecoder(nn.Module):
    def __init__(self, neighbors: int = 10, max_time: int = 8, max_branch: int = 30, n_heads: int = 8, dim: int = 256, variable_cost: bool = False):
        super().__init__()
        self.neighbors = int(neighbors)
        self.nheads = int(n_heads)
        self.time = int(max_time)
        self.branch = int(max_branch)
        self.environment_decoder = CrossAttention(n_heads, dim)
        self.ego_condition_decoder = CrossAttention(n_heads, dim)
        self.time_embed = nn.Embedding(max_time, dim)
        self.ego_traj_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, dim))
        self.agent_traj_decoder = AgentDecoder(max_time, max_branch, dim * 2)
        self.ego_traj_decoder = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, max_time * 10 * 3))
        self.scorer = ScoreDecoder(variable_cost)
        self.register_buffer("casual_mask", self.generate_casual_mask())
        self.register_buffer("time_index", torch.arange(max_time).repeat(max_branch, 1))

    def pooling_trajectory(self, trajectory_tree: torch.Tensor) -> torch.Tensor:
        B, M, T, D = trajectory_tree.shape
        if T < 10:
            return torch.max(trajectory_tree, dim=2, keepdim=True).values
        pad = (-T) % 10
        if pad:
            trajectory_tree = F.pad(trajectory_tree, (0, 0, 0, pad))
            T = T + pad
        trajectory_tree = torch.reshape(trajectory_tree, (B, M, T // 10, 10, D))
        return torch.max(trajectory_tree, dim=-2)[0]

    def generate_casual_mask(self) -> torch.Tensor:
        time_mask = torch.tril(torch.ones(self.time, self.time))
        mask = torch.zeros(self.branch * self.time, self.branch * self.time)
        for i in range(self.branch):
            mask[i * self.time : (i + 1) * self.time, i * self.time : (i + 1) * self.time] = time_mask
        return mask

    def forward(self, encoder_outputs: Mapping[str, torch.Tensor], ego_traj_inputs: torch.Tensor, agents_states: torch.Tensor, timesteps: int, candidate_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_states = agents_states[:, : self.neighbors, -1]
        encoding, encoding_mask = encoder_outputs["encoding"], encoder_outputs["mask"]
        ego_traj_ori_encoding = self.ego_traj_encoder(ego_traj_inputs)
        branch_t = min(max(int(timesteps), 1), ego_traj_ori_encoding.shape[2]) - 1
        branch_embedding = ego_traj_ori_encoding[:, :, branch_t]
        time_embedding = self.time_embed(self.time_index)
        tree_embedding = time_embedding[None] + branch_embedding[:, :, None, :]
        raw_ego_traj_mask = torch.ne(ego_traj_inputs.abs().sum(-1), 0)
        if candidate_valid is not None:
            # A valid full-stop branch may be exactly zero for all 80 steps in
            # the ego frame.  Keep it addressable in the decoder; invalid/padded
            # branches remain fully masked by the explicit proposal mask.
            raw_ego_traj_mask = torch.where(
                candidate_valid.bool()[:, :, None],
                torch.ones_like(raw_ego_traj_mask),
                torch.zeros_like(raw_ego_traj_mask),
            )
        # Public DTPP summarizes each 1 s / 10-step chunk by max pooling before
        # ego-conditioned attention.  The previous adapter used strided samples,
        # which changed the model and made a single noisy step dominate masks.
        ego_tree_for_attn = self.pooling_trajectory(ego_traj_ori_encoding)
        mask_float = raw_ego_traj_mask.float().unsqueeze(-1)
        pooled_mask = self.pooling_trajectory(mask_float).squeeze(-1) > 0.5
        ego_traj_mask_3d = pooled_mask[:, :, : self.time]
        ego_tree_for_attn = ego_tree_for_attn[:, :, : self.time]
        if ego_traj_mask_3d.shape[2] < self.time:
            pad_t = self.time - ego_traj_mask_3d.shape[2]
            ego_traj_mask_3d = F.pad(ego_traj_mask_3d, (0, pad_t), value=False)
            ego_tree_for_attn = F.pad(ego_tree_for_attn, (0, 0, 0, pad_t), value=0.0)
        ego_traj_mask = torch.reshape(ego_traj_mask_3d, (ego_traj_mask_3d.shape[0], -1)).bool()
        # MultiheadAttention returns NaNs when a query row has every key masked.
        # Padded/invalid candidate branches are later removed by candidate_valid,
        # so let those query rows attend normally and mask only valid query rows.
        env_allowed = ego_traj_mask[:, :, None] & encoding_mask.logical_not()[:, None, :]
        env_mask = torch.where(env_allowed, 0.0, -1e9)
        env_mask = torch.where(ego_traj_mask[:, :, None], env_mask, torch.zeros_like(env_mask))
        env_mask = env_mask.repeat_interleave(self.nheads, dim=0)
        causal = self.casual_mask.to(device=ego_traj_inputs.device, dtype=torch.bool)[None]
        ego_key_valid = ego_traj_mask[:, None, :]
        ego_allowed = ego_traj_mask[:, :, None] & ego_key_valid & causal
        ego_condition_mask = torch.where(ego_allowed, 0.0, -1e9)
        ego_condition_mask = torch.where(ego_traj_mask[:, :, None], ego_condition_mask, torch.zeros_like(ego_condition_mask))
        ego_condition_mask = ego_condition_mask.repeat_interleave(self.nheads, dim=0)
        ego_flat = torch.reshape(ego_tree_for_attn, (ego_tree_for_attn.shape[0], -1, ego_tree_for_attn.shape[-1]))
        agents_trajectories = []
        for i in range(self.neighbors):
            query = encoding[:, i + 1, None, None] + tree_embedding
            query = torch.reshape(query, (query.shape[0], -1, query.shape[-1]))
            env_dec = self.environment_decoder(query, encoding, encoding, env_mask)
            ego_dec = self.ego_condition_decoder(query, ego_flat, ego_flat, ego_condition_mask)
            dec = torch.cat([env_dec, ego_dec], dim=-1)
            agents_trajectories.append(self.agent_traj_decoder(dec, current_states[:, i]))
        agents_trajectories = torch.stack(agents_trajectories, dim=2)
        scores, weights = self.scorer(
            ego_traj_inputs, encoding[:, 0], agents_trajectories, current_states, timesteps, candidate_valid=candidate_valid
        )
        ego_reg = self.ego_traj_decoder(encoding[:, 0]).reshape(encoding.shape[0], self.time * 10, 3)
        return agents_trajectories, scores, ego_reg, weights


class COWPDTPP(nn.Module):
    def __init__(self, neighbors: int = 10, max_branch: int = 30, variable_cost: bool = False):
        super().__init__()
        self.neighbors = int(neighbors)
        self.max_branch = int(max_branch)
        self.encoder = DTPPEncoder()
        self.decoder = DTPPDecoder(neighbors=neighbors, max_branch=max_branch, variable_cost=variable_cost)

    def forward(self, inputs: Mapping[str, torch.Tensor], ego_traj_tree: torch.Tensor, timesteps: int = 80, candidate_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(inputs)
        return self.decoder(
            enc, ego_traj_tree, inputs["neighbor_agents_past"], timesteps, candidate_valid=candidate_valid
        )

    def score_candidates(self, inputs: Mapping[str, torch.Tensor], ego_traj_tree: torch.Tensor, candidate_valid: torch.Tensor, timesteps: int = 80) -> torch.Tensor:
        _, scores, _, _ = self.forward(inputs, ego_traj_tree, timesteps=timesteps, candidate_valid=candidate_valid)
        return torch.where(candidate_valid, scores, torch.full_like(scores, -1e9))


def dtpp_loss(model: COWPDTPP, inputs: Mapping[str, torch.Tensor], ego_traj_tree: torch.Tensor, candidate_valid: torch.Tensor, best_idx: torch.Tensor, ego_future_xy: torch.Tensor, ego_future_valid: torch.Tensor, neighbors_future_xy: torch.Tensor, neighbors_future_valid: torch.Tensor, timesteps: int = 80) -> tuple[torch.Tensor, dict[str, float]]:
    neighbors_pred, scores, ego_reg, weights = model(
        inputs, ego_traj_tree, timesteps=timesteps, candidate_valid=candidate_valid
    )
    # Loss-side FP32 keeps AMP forward speed while preventing SmoothL1/score
    # arithmetic from silently producing non-finite batches.
    neighbors_pred = neighbors_pred.float()
    scores = scores.float()
    ego_reg = ego_reg.float()
    weights = weights.float()
    ego_future_xy = ego_future_xy.float()
    neighbors_future_xy = neighbors_future_xy.float()
    B = scores.shape[0]
    sample_valid = candidate_valid.any(dim=1) & ego_future_valid.any(dim=1)
    if not bool(sample_valid.any()):
        zero = scores.sum() * 0.0 + neighbors_pred.sum() * 0.0 + ego_reg.sum() * 0.0 + weights.sum() * 0.0
        return zero, {"plannerADE": float("nan"), "valid_samples": 0.0}

    scores_masked = torch.where(candidate_valid, scores, torch.full_like(scores, -1e9))
    row_all = torch.arange(B, device=scores.device)
    rows = row_all[sample_valid]
    target = best_idx[sample_valid].long()
    ce = F.cross_entropy(scores_masked[sample_valid], target)

    pred = neighbors_pred[rows, target]
    T = min(pred.shape[2], neighbors_future_xy.shape[2])
    pred_xy = pred[:, :, :T, :2]
    gt_xy = neighbors_future_xy[sample_valid, :, :T, :2]
    valid = neighbors_future_valid[sample_valid, :, :T].float()
    cmp_loss = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none").sum(-1)
    cmp_loss = (cmp_loss * valid).sum() / valid.sum().clamp_min(1.0)

    ego_T = min(ego_reg.shape[1], ego_future_xy.shape[1])
    ego_valid = ego_future_valid[sample_valid, :ego_T].float()
    reg = F.smooth_l1_loss(ego_reg[sample_valid, :ego_T, :2], ego_future_xy[sample_valid, :ego_T], reduction="none").sum(-1)
    reg = (reg * ego_valid).sum() / ego_valid.sum().clamp_min(1.0)
    wreg = torch.square(weights[sample_valid]).mean()
    loss = ce + cmp_loss + 0.1 * reg + 0.01 * wreg

    sel = torch.argmax(scores_masked[sample_valid], dim=1)
    plan = ego_traj_tree[sample_valid][torch.arange(sel.shape[0], device=scores.device), sel, :, :2]
    pt = min(plan.shape[1], ego_future_xy.shape[1])
    valid_ego = ego_future_valid[sample_valid, :pt].float()
    pde = torch.linalg.norm((plan[:, :pt] - ego_future_xy[sample_valid, :pt]) * valid_ego[:, :, None], dim=-1)
    valid_score_values = scores[candidate_valid & sample_valid[:, None]]
    score_abs_max = valid_score_values.abs().max() if valid_score_values.numel() else scores.sum() * 0.0
    metrics = {
        "plannerADE": float((pde.sum() / valid_ego.sum().clamp_min(1.0)).detach().cpu()),
        "score_ce": float(ce.detach().cpu()),
        "neighbor_cmp": float(cmp_loss.detach().cpu()),
        "ego_reg": float(reg.detach().cpu()),
        "weight_reg": float(wreg.detach().cpu()),
        "score_abs_max": float(score_abs_max.detach().cpu()),
        "weight_max": float(weights[sample_valid].abs().max().detach().cpu()),
        "valid_samples": float(sample_valid.float().sum().detach().cpu()),
    }
    return loss, metrics
