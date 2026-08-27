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
    def __init__(self, dim: int = 256):
        super().__init__()
        self.motion = nn.LSTM(8, dim, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, dim, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        traj, _ = self.motion(inputs[:, :, :8])
        typ = self.type_emb(inputs[:, -1, 8].long().clamp(0, 3))
        return traj[:, -1] + typ


class LaneEncoder(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, dim))
        self.position_encode = PositionalEncoding(dim, max_len=100)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[..., 6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].long().clamp(0, 3))
        left_type = self.left_type(inputs[..., 11].long().clamp(0, 10))
        right_type = self.right_type(inputs[..., 12].long().clamp(0, 10))
        traffic_light = self.traffic_light_type(inputs[..., 13].long().clamp(0, 8))
        interpolating = self.interpolating(inputs[..., 14].long().clamp(0, 1))
        stop_sign = self.stop_sign(inputs[..., 15].long().clamp(0, 1))
        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
        return self.position_encode(self.pointnet(lane_embedding))


class CrosswalkEncoder(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.point_net(inputs)


class FutureEncoder(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, dim))
        self.type_emb = nn.Embedding(4, dim, padding_idx=0)

    def state_process(self, trajs: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1.0e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        return torch.cat([trajs, theta, v, size], dim=-1)

    def forward(self, trajs: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
        x = self.state_process(trajs, current_states)
        x = self.mlp(x.detach())
        typ = self.type_emb(current_states[:, :, None, 8].long().clamp(0, 3))
        return torch.max(x, dim=-2).values + typ


class GMMPredictor(nn.Module):
    def __init__(self, future_len: int, dim: int = 256):
        super().__init__()
        self.future_len = int(future_len)
        self.gaussian = nn.Sequential(nn.Linear(dim, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self.future_len * 4))
        self.score = nn.Sequential(nn.Linear(dim, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, _ = x.shape
        res = self.gaussian(x).view(B, M, self.future_len, 4)
        score = self.score(x).squeeze(-1)
        return res, score


class SelfTransformer(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.self_attention(x, x, x, key_padding_mask=mask)
        y = self.norm_1(y + x)
        return self.norm_2(self.ffn(y) + y)


class CrossTransformer(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        # Match the public GameFormer CrossTransformer: cross-attention is
        # normalized directly here (there is no query residual before norm_1).
        # The previous COWP adapter added ``+ query``, which changes the source
        # architecture and lets the recursive interaction decoder repeatedly
        # amplify its own query state.
        y = self.norm_1(y)
        return self.norm_2(self.ffn(y) + y)


class InitialDecoder(nn.Module):
    def __init__(self, modalities: int, neighbors: int, future_len: int, dim: int = 256):
        super().__init__()
        self.modalities = int(modalities)
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors + 1, dim)
        self.query_encoder = CrossTransformer(dim)
        self.predictor = GMMPredictor(future_len, dim)
        self.register_buffer("modal", torch.arange(modalities).long())
        self.register_buffer("agent", torch.arange(neighbors + 1).long())

    def forward(self, idx: int, current_state: torch.Tensor, encoding: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[idx])
        query = encoding[:, None, idx] + modal_query + agent_query[None]
        content = self.query_encoder(query, encoding, encoding, mask)
        pred, scores = self.predictor(content)
        pred[..., :2] += current_state[:, None, None, :2]
        return content, pred, scores


class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder: FutureEncoder, future_len: int, dim: int = 256):
        super().__init__()
        self.interaction_encoder = SelfTransformer(dim)
        self.query_encoder = CrossTransformer(dim)
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len, dim)

    def forward(self, idx: int, current_states: torch.Tensor, actors: torch.Tensor, scores: torch.Tensor, last_content: torch.Tensor, encoding: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, M, T, _ = actors.shape
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        # Source GameFormer averages the probability-weighted modal features over
        # the modality axis.  The V5 adapter used ``sum`` instead of ``mean``;
        # with M=6 this changes the recursive interaction-feature scale by 6x
        # relative to the published implementation and compounds instability.
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)
        interaction = self.interaction_encoder(futures, mask[:, :N])
        encoding2 = torch.cat([interaction, encoding], dim=1)
        mask2 = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask2[:, idx] = True
        query = last_content + multi_futures[:, idx]
        content = self.query_encoder(query, encoding2, encoding2, mask2)
        pred, new_scores = self.decoder(content)
        pred[..., :2] += current_states[:, idx, None, None, :2]
        return content, pred, new_scores


class GameFormerEncoder(nn.Module):
    def __init__(self, neighbors_to_predict: int, layers: int = 6, dim: int = 256):
        super().__init__()
        self.neighbors = int(neighbors_to_predict)
        self.agent_encoder = AgentEncoder(dim)
        self.ego_encoder = AgentEncoder(dim)
        self.lane_encoder = LaneEncoder(dim)
        self.crosswalk_encoder = CrosswalkEncoder(dim)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4, activation=F.gelu, dropout=0.1, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def segment_map(
        self, map_tensor: torch.Tensor, map_encoding: torch.Tensor,
        point_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride = 10
        B, N_e, N_p, D = map_encoding.shape
        if N_p % stride != 0:
            raise ValueError(f"GameFormer map points must be divisible by {stride}, got {N_p}")
        if point_valid is None:
            # Backward-compatible fallback only.  The COWP/WOMD adapter passes
            # source validity explicitly because a valid local point can be 0.
            point_valid = ~torch.eq(map_tensor, 0).all(dim=-1)
        point_valid = point_valid.bool()
        nseg = N_p // stride
        enc = map_encoding.reshape(B, N_e, nseg, stride, D)
        valid = point_valid.reshape(B, N_e, nseg, stride)
        floor = torch.finfo(map_encoding.dtype).min
        enc = enc.masked_fill(~valid[..., None], floor)
        pooled = enc.max(dim=3).values
        segment_valid = valid.any(dim=3)
        pooled = torch.where(segment_valid[..., None], pooled, torch.zeros_like(pooled))
        return pooled.reshape(B, -1, D), (~segment_valid).reshape(B, -1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ego = inputs["ego_state"]
        neighbors = inputs["neighbors_state"]
        actors = torch.cat([ego.unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_valid = inputs.get("actors_valid")
        if actors_valid is not None:
            actors_mask = ~actors_valid[:, :, -1].bool()
        else:
            actors_mask = torch.eq(actors[:, :, -1], 0).all(dim=-1)
        # TransformerEncoder can emit NaNs when every token of a row is masked.
        # Keep a zero-valued SDC anchor token addressable only for the pathological
        # invalid-SDC row; normal valid rows are bit-for-bit unchanged.
        invalid_anchor = actors_mask[:, 0].clone()
        if bool(invalid_anchor.any()):
            encoded_actors = encoded_actors.clone()
            encoded_actors[invalid_anchor, 0] = 0.0
            actors_mask = actors_mask.clone()
            actors_mask[:, 0] = False
        map_lanes = inputs["map_lanes"]
        map_crosswalks = inputs["map_crosswalks"]
        encoded_map_lanes = self.lane_encoder(map_lanes)
        encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)
        encodings, masks = [], []
        N = self.neighbors + 1
        for i in range(N):
            lane_valid = inputs.get("map_lanes_valid")
            cross_valid = inputs.get("map_crosswalks_valid")
            lanes, lanes_mask = self.segment_map(
                map_lanes[:, i], encoded_map_lanes[:, i], None if lane_valid is None else lane_valid[:, i]
            )
            crosswalks, cross_mask = self.segment_map(
                map_crosswalks[:, i], encoded_map_crosswalks[:, i], None if cross_valid is None else cross_valid[:, i]
            )
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, cross_mask], dim=1)
            encodings.append(self.fusion_encoder(fusion_input, src_key_padding_mask=mask))
            masks.append(mask)
        return {"actors": actors, "encodings": torch.stack(encodings, dim=1), "masks": torch.stack(masks, dim=1)}


class GameFormerDecoder(nn.Module):
    def __init__(self, modalities: int, future_len: int, neighbors_to_predict: int, levels: int = 4, dim: int = 256):
        super().__init__()
        self.levels = int(levels)
        self.neighbors = int(neighbors_to_predict)
        future_encoder = FutureEncoder(dim)
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len, dim)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_len, dim) for _ in range(levels)])

    def forward(self, encoder_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        N = self.neighbors + 1
        current_states = encoder_inputs["actors"][:, :, -1]
        encodings, masks = encoder_inputs["encodings"], encoder_inputs["masks"]
        results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([r[0] for r in results], dim=1)
        last_level = torch.stack([r[1] for r in results], dim=1)
        last_scores = torch.stack([r[2] for r in results], dim=1)
        outputs["level_0_interactions"] = last_level
        outputs["level_0_scores"] = last_scores
        for k in range(1, self.levels + 1):
            dec = self.interaction_stage[k - 1]
            results = [dec(i, current_states[:, :N], last_level, last_scores, last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
            last_content = torch.stack([r[0] for r in results], dim=1)
            last_level = torch.stack([r[1] for r in results], dim=1)
            last_scores = torch.stack([r[2] for r in results], dim=1)
            outputs[f"level_{k}_interactions"] = last_level
            outputs[f"level_{k}_scores"] = last_scores
        return outputs


class COWPGameFormer(nn.Module):
    def __init__(self, modalities: int = 6, neighbors_to_predict: int = 10, future_len: int = 80, encoder_layers: int = 6, decoder_levels: int = 4):
        super().__init__()
        self.neighbors_to_predict = int(neighbors_to_predict)
        self.future_len = int(future_len)
        self.encoder = GameFormerEncoder(neighbors_to_predict, encoder_layers)
        self.decoder = GameFormerDecoder(modalities, future_len, neighbors_to_predict, decoder_levels)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decoder(self.encoder(inputs))

    def final_level(self, outputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        levels = sorted(int(k.split("_")[1]) for k in outputs if k.startswith("level_") and k.endswith("_interactions"))
        k = levels[-1]
        return outputs[f"level_{k}_interactions"], outputs[f"level_{k}_scores"]

    def score_candidates(self, inputs: Mapping[str, torch.Tensor], candidates: torch.Tensor, candidate_valid: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(inputs)
        trajs, mode_scores = self.final_level(outputs)
        ego_modes = trajs[:, 0, :, :, :2]
        B, K, T, _ = candidates.shape
        Tm = min(T, ego_modes.shape[2])
        diff = candidates[:, :, None, :Tm, :2] - ego_modes[:, None, :, :Tm, :2]
        ade = torch.linalg.norm(diff, dim=-1).mean(dim=-1)
        mode_bonus = mode_scores[:, 0, :].softmax(-1).clamp_min(1e-6).log()
        score = -(ade - 0.05 * mode_bonus[:, None, :]).min(dim=-1).values
        return torch.where(candidate_valid, score, torch.full_like(score, -1e9))


def imitation_loss(gmm: torch.Tensor, scores: torch.Tensor, gt_xy: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    # gmm [B,N,M,T,4], scores [B,N,M], gt [B,N,T,2], valid [B,N,T]
    # Keep regression/NLL arithmetic in FP32 under AMP.  FP16 squared global or
    # even moderately large local residuals can overflow before GradScaler sees
    # the loss; casting here is differentiable and leaves model matmuls in AMP.
    gmm = gmm.float()
    scores = scores.float()
    gt_xy = gt_xy.float()
    valid_f = valid.float()
    agent_valid = valid.any(dim=-1)
    denom_agent = valid_f.sum(dim=-1).clamp_min(1.0)
    # Sanitize the target *before* subtraction.  Masking a NaN residual after
    # the subtraction is too late for some fused/autograd kernels.
    gt_modes = gt_xy[:, :, None].expand(-1, -1, gmm.shape[2], -1, -1)
    safe_target = torch.where(valid[:, :, None, :, None], gt_modes, gmm[..., :2].detach())
    safe_delta = gmm[..., :2] - safe_target
    dist = torch.linalg.norm(safe_delta, dim=-1)
    mode_ade = dist.sum(dim=-1) / denom_agent[:, :, None]
    mode_ade = torch.where(agent_valid[:, :, None], mode_ade, torch.zeros_like(mode_ade))
    scene_mode_ade = mode_ade.sum(dim=1) / agent_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    best_mode = torch.argmin(scene_mode_ade, dim=-1)
    B, N, M, T, _ = gmm.shape
    gather_idx = best_mode[:, None, None, None, None].expand(B, N, 1, T, 4)
    best = torch.gather(gmm, 2, gather_idx).squeeze(2)
    safe_best_target = torch.where(valid[..., None], gt_xy, best[..., :2].detach())
    dx = safe_best_target[..., 0] - best[..., 0]
    dy = safe_best_target[..., 1] - best[..., 1]
    log_std_x = torch.clamp(best[..., 2], -2, 2)
    log_std_y = torch.clamp(best[..., 3], -2, 2)
    loss_t = log_std_x + log_std_y + 0.5 * ((dx / torch.exp(log_std_x)) ** 2 + (dy / torch.exp(log_std_y)) ** 2)
    reg = (loss_t * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    # Original GameFormer uses one scene-level mode.  Mask invalid padded agents so
    # nonexistent neighbors do not train the mode classifier.
    ce_target = best_mode[:, None].expand(-1, N)
    ce_per_agent = F.cross_entropy(scores.permute(0, 2, 1), ce_target, label_smoothing=0.2, reduction="none")
    ce = (ce_per_agent * agent_valid.float()).sum() / agent_valid.float().sum().clamp_min(1.0)
    future = best[..., :2]
    return reg + ce, future, {"gmm_nll": reg.detach(), "mode_ce": ce.detach()}


def interaction_loss(trajectories: torch.Tensor, last_trajectories: torch.Tensor, neighbors_valid: torch.Tensor) -> torch.Tensor:
    trajectories = trajectories.float()
    last_trajectories = last_trajectories.float()
    B, N, M, T, _ = trajectories.shape
    if N <= 1:
        return trajectories.sum() * 0.0
    neighbors_mask = neighbors_valid.logical_not()
    vals = []
    for t in range(T):
        ego_p = trajectories[:, 0, :, t, :2]
        last_neighbors_p = last_trajectories[:, 1:, :, t, :2].reshape(B, -1, 2)
        dist_to_ego = torch.cdist(ego_p, last_neighbors_p)
        n_mask = neighbors_mask.unsqueeze(-1).expand(-1, -1, M).reshape(B, 1, -1)
        dist_to_ego = torch.masked_fill(dist_to_ego, n_mask, 1000.0)
        vals.append((1.0 / (dist_to_ego.min(dim=-1).values + 1.0) * (dist_to_ego.min(dim=-1).values < 3)).sum(-1).mean())
    return torch.stack(vals).mean()


def gameformer_loss(outputs: Mapping[str, torch.Tensor], ego_future_xy: torch.Tensor, ego_future_valid: torch.Tensor, neighbors_future_xy: torch.Tensor, neighbors_future_valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    gt = torch.cat([ego_future_xy[:, None], neighbors_future_xy], dim=1)
    valid = torch.cat([ego_future_valid[:, None], neighbors_future_valid], dim=1)
    # Never seed the differentiable zero from raw labels: an invalid NaN label
    # would make ``NaN * 0`` and contaminate an otherwise mask-safe batch.
    first_output = next(v for v in outputs.values() if torch.is_tensor(v))
    total = first_output.sum() * 0.0
    future = None
    metric_tensors: dict[str, torch.Tensor] = {}
    levels = len([k for k in outputs if k.endswith("_interactions")])
    for k in range(levels):
        traj = outputs[f"level_{k}_interactions"]
        scores = outputs[f"level_{k}_scores"]
        il, future, parts = imitation_loss(traj, scores, gt, valid)
        total = total + il
        metric_tensors[f"level{k}_gmm_nll"] = parts["gmm_nll"]
        metric_tensors[f"level{k}_mode_ce"] = parts["mode_ce"]
        if k >= 1:
            nvalid = neighbors_future_valid.any(dim=-1) if neighbors_future_valid.numel() else torch.zeros(gt.shape[0], 0, device=gt.device, dtype=torch.bool)
            inter = interaction_loss(traj, outputs[f"level_{k-1}_interactions"], nvalid)
            total = total + 0.1 * inter
            metric_tensors[f"level{k}_interaction"] = inter.detach()
    metrics: dict[str, float] = {k: float(v.detach().cpu()) for k, v in metric_tensors.items()}
    if future is not None:
        valid_f = ego_future_valid.float()
        ego_target_safe = torch.where(ego_future_valid[..., None], ego_future_xy, future[:, 0].detach())
        ego_dist = torch.linalg.norm(future[:, 0] - ego_target_safe, dim=-1)
        ego_dist = torch.where(ego_future_valid, ego_dist, torch.zeros_like(ego_dist))
        metrics["plannerADE"] = float((ego_dist.sum() / valid_f.sum().clamp_min(1.0)).detach().cpu())
        sample_valid = ego_future_valid.any(dim=1)
        metrics["valid_samples"] = float(sample_valid.float().sum().detach().cpu())
        if bool(sample_valid.any()):
            last_idx = valid_f.sum(dim=1).long().clamp_min(1) - 1
            rows = torch.arange(ego_dist.shape[0], device=ego_dist.device)
            metrics["plannerFDE"] = float(ego_dist[rows[sample_valid], last_idx[sample_valid]].mean().detach().cpu())
    metrics.setdefault("valid_samples", float(ego_future_valid.any(dim=1).float().sum().detach().cpu()))
    return total, metrics
