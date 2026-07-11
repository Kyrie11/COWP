from __future__ import annotations

import torch
from torch import nn


class GraphEncoder(nn.Module):
    """Burden-oriented heterogeneous graph encoder.

    The paper describes a graph with typed ego/agent/candidate/conflict nodes,
    candidate-conditioned edges, a natural-vs-conditioned dual edge view and a
    compact conflict query token.  This module keeps the same padded tensor API as
    the original implementation, but makes those switches real:

    * typed edge messages are injected before the transformer;
    * candidate-to-agent and agent-to-candidate messages use trajectory proximity;
    * candidate/agent-to-conflict messages use conflict-region distance;
    * a learned conflict-query token summarizes high-conflict evidence;
    * ``use_dual_edge=False`` removes the natural/conditioned edge-type split;
    * ``use_conflict_query=False`` removes the query token and its aggregation.

    The implementation intentionally avoids PyG/DGL so the repository remains a
    lightweight WOMD/Waymax project with only PyTorch as the model dependency.
    """

    def __init__(
        self,
        d_state: int = 11,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        *,
        use_typed_edges: bool = True,
        use_dual_edge: bool = True,
        use_conflict_query: bool = True,
        edge_distance_scale_m: float = 12.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.use_typed_edges = bool(use_typed_edges)
        self.use_dual_edge = bool(use_dual_edge)
        self.use_conflict_query = bool(use_conflict_query)
        self.edge_distance_scale_m = float(edge_distance_scale_m)

        self.agent_proj = nn.Linear(d_state, d_model)
        self.candidate_proj = nn.Linear(7, d_model)
        self.conflict_proj = nn.Linear(8, d_model)
        self.type_embed = nn.Embedding(5, d_model)

        # Edge-type ids: generic, conditioned candidate-agent, natural candidate-agent,
        # candidate-conflict, agent-conflict, ego-candidate, conflict-query, padding.
        self.edge_type_embed = nn.Embedding(8, d_model)
        self.edge_feat_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.agent_update = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.candidate_update = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.conflict_update = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.query_update = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.conflict_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.conflict_query, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _history_mean(agent_history: torch.Tensor) -> torch.Tensor:
        """Return one state vector per agent from temporal history.

        Avoid Python-side tensor control flow such as ``if empty.any()``.  Some
        PyTorch Inductor/Triton versions miscompile bool reductions under
        ``torch.compile``; always-on ``torch.where`` is equivalent and stable.
        """
        if agent_history.ndim == 4:
            if agent_history.shape[-1] >= 11:
                hist_valid = torch.nan_to_num(agent_history[..., 10:11].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                clean_history = torch.nan_to_num(agent_history.float(), nan=0.0, posinf=0.0, neginf=0.0)
                valid_sum = hist_valid.sum(dim=2)
                x_agent = (clean_history * hist_valid).sum(dim=2) / valid_sum.clamp_min(1.0)
                fallback = clean_history.mean(dim=2)
                empty = valid_sum.squeeze(-1) <= 0
                return torch.where(empty.unsqueeze(-1), fallback, x_agent)
            return torch.nan_to_num(agent_history.float(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=2)
        return torch.nan_to_num(agent_history.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _candidate_agent_features(self, x_agent: torch.Tensor, candidate_traj: torch.Tensor) -> torch.Tensor:
        """Return [B,K,N,8] edge features for candidate-conditioned interactions."""
        B, K, T, _ = candidate_traj.shape
        N = x_agent.shape[1]
        cand_xy = candidate_traj[..., :2]
        cand_v = candidate_traj[..., 3:5]
        agent_xy = x_agent[..., 0:2]
        # In this repository d_state uses [x,y,z,length,width,height,heading,vx,vy,speed,valid].
        agent_v = x_agent[..., 7:9] if x_agent.shape[-1] >= 9 else torch.zeros(B, N, 2, device=x_agent.device, dtype=x_agent.dtype)
        delta = cand_xy[:, :, None, :, :] - agent_xy[:, None, :, None, :]
        dist = torch.linalg.norm(delta, dim=-1)
        min_dist, argmin = dist.min(dim=-1)
        # Gather closest-time relative position and velocity summaries.
        idx = argmin[..., None, None].expand(B, K, N, 1, 2)
        closest_delta = torch.gather(delta, 3, idx).squeeze(3)
        cand_speed = torch.linalg.norm(cand_v, dim=-1).mean(dim=-1)
        agent_speed = torch.linalg.norm(agent_v, dim=-1)
        rel_speed = cand_speed[:, :, None] - agent_speed[:, None, :]
        progress = torch.linalg.norm(cand_xy[:, :, -1] - cand_xy[:, :, 0], dim=-1)
        tta_proxy = argmin.float() / max(T - 1, 1)
        scale = max(self.edge_distance_scale_m, 1e-3)
        prox = torch.exp(-min_dist / scale)
        return torch.stack(
            [
                closest_delta[..., 0] / scale,
                closest_delta[..., 1] / scale,
                min_dist / scale,
                prox,
                rel_speed / 15.0,
                tta_proxy,
                progress[:, :, None].expand_as(min_dist) / 60.0,
                torch.ones_like(min_dist),
            ],
            dim=-1,
        )

    def _conflict_features(self, xy: torch.Tensor, conflict_regions: torch.Tensor) -> torch.Tensor:
        """Return edge features from a token trajectory/current xy to conflict nodes.

        ``xy`` is [B,N,2] or [B,K,T,2]; output is [B,N,C,8] or [B,K,C,8].
        """
        center = conflict_regions[..., 1:3]
        radius = conflict_regions[..., 3].clamp_min(0.1)
        if xy.ndim == 4:
            delta = xy[:, :, None, :, :] - center[:, None, :, None, :]
            dist_t = torch.linalg.norm(delta, dim=-1)
            min_dist, argmin = dist_t.min(dim=-1)
            closest = torch.gather(delta, 3, argmin[..., None, None].expand(*argmin.shape, 1, 2)).squeeze(3)
        else:
            closest = xy[:, :, None, :] - center[:, None, :, :]
            min_dist = torch.linalg.norm(closest, dim=-1)
            argmin = torch.zeros_like(min_dist)
        scale = max(self.edge_distance_scale_m, 1e-3)
        radius_b = radius[:, None, :].expand_as(min_dist)
        prox = torch.exp(-torch.relu(min_dist - radius_b) / scale)
        tta = argmin.float() / max(xy.shape[-2] - 1, 1) if xy.ndim == 4 else torch.zeros_like(min_dist)
        return torch.stack(
            [
                closest[..., 0] / scale,
                closest[..., 1] / scale,
                min_dist / scale,
                prox,
                radius_b / scale,
                tta,
                torch.ones_like(min_dist),
                torch.ones_like(min_dist),
            ],
            dim=-1,
        )

    @staticmethod
    def _aggregate(edge_h: torch.Tensor, edge_mask: torch.Tensor, dim: int) -> torch.Tensor:
        w = edge_mask.float().unsqueeze(-1)
        denom = w.sum(dim=dim).clamp_min(1.0)
        return (edge_h * w).sum(dim=dim) / denom

    def _edge_type_bias(self, edge_ndim_without_channel: int, edge_type: int) -> torch.Tensor:
        """Return a broadcastable edge-type embedding without allocating a full LongTensor.

        The previous implementation called ``torch.full(edge_shape, edge_type)``
        on every forward pass before an embedding lookup.  With large batches and
        PyTorch CUDA expandable segments this can trigger allocator internal
        assertions even though the model tensors are valid.  Index the embedding
        weight directly and rely on broadcasting instead.
        """
        return self.edge_type_embed.weight[int(edge_type)].view(*([1] * int(edge_ndim_without_channel)), -1)

    def _inject_edge_messages(
        self,
        z_agent: torch.Tensor,
        agent_mask: torch.Tensor,
        x_agent: torch.Tensor,
        z_cand: torch.Tensor | None,
        candidate_traj: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        z_conf: torch.Tensor | None,
        conflict_regions: torch.Tensor | None,
        conflict_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_typed_edges:
            return z_agent, z_cand, z_conf, None

        query_msg = None
        if z_cand is not None and candidate_traj is not None:
            cm = candidate_mask.bool() if candidate_mask is not None else torch.ones(z_cand.shape[:2], device=z_cand.device, dtype=torch.bool)
            edge_feat = self._candidate_agent_features(x_agent, candidate_traj)
            pair_mask = cm[:, :, None] & agent_mask.bool()[:, None, :]
            edge_type = 1 if self.use_dual_edge else 0
            cond_h = self.edge_feat_proj(edge_feat) + self._edge_type_bias(edge_feat.ndim - 1, edge_type)
            z_cand = z_cand + self.candidate_update(self._aggregate(cond_h, pair_mask, dim=2))
            z_agent = z_agent + self.agent_update(self._aggregate(cond_h, pair_mask, dim=1))
            if self.use_dual_edge:
                # Natural edge view: candidate-independent proximity/slack evidence.  This
                # separates "what the agent was likely to do" from "what the agent must
                # do under a candidate" without requiring extra graph libraries.
                nat_feat = edge_feat.clone()
                nat_feat[..., 4] = 0.0  # remove candidate relative-speed conditioning
                nat_h = self.edge_feat_proj(nat_feat) + self._edge_type_bias(edge_feat.ndim - 1, 2)
                z_agent = z_agent + 0.5 * self.agent_update(self._aggregate(nat_h, pair_mask, dim=1))

        if z_conf is not None and conflict_regions is not None:
            fm = conflict_mask.bool() if conflict_mask is not None else torch.ones(z_conf.shape[:2], device=z_conf.device, dtype=torch.bool)
            # Agent-current to conflict region.
            agent_cf = self._conflict_features(x_agent[..., :2], conflict_regions)
            agent_cf_mask = agent_mask.bool()[:, :, None] & fm[:, None, :]
            agent_h = self.edge_feat_proj(agent_cf) + self._edge_type_bias(agent_cf.ndim - 1, 4)
            z_agent = z_agent + self.agent_update(self._aggregate(agent_h, agent_cf_mask, dim=2))
            z_conf = z_conf + self.conflict_update(self._aggregate(agent_h, agent_cf_mask, dim=1))
            query_msg = self._aggregate(agent_h, agent_cf_mask, dim=(1, 2)) if False else None

            if z_cand is not None and candidate_traj is not None:
                cand_cf = self._conflict_features(candidate_traj[..., :2], conflict_regions)
                cm = candidate_mask.bool() if candidate_mask is not None else torch.ones(z_cand.shape[:2], device=z_cand.device, dtype=torch.bool)
                cand_cf_mask = cm[:, :, None] & fm[:, None, :]
                cand_h = self.edge_feat_proj(cand_cf) + self._edge_type_bias(cand_cf.ndim - 1, 3)
                z_cand = z_cand + self.candidate_update(self._aggregate(cand_h, cand_cf_mask, dim=2))
                z_conf = z_conf + self.conflict_update(self._aggregate(cand_h, cand_cf_mask, dim=1))
                # Query uses the strongest candidate-conflict messages.
                w = cand_cf_mask.float().unsqueeze(-1)
                denom = w.sum(dim=(1, 2)).clamp_min(1.0)
                query_msg = (cand_h * w).sum(dim=(1, 2)) / denom
            else:
                w = agent_cf_mask.float().unsqueeze(-1)
                denom = w.sum(dim=(1, 2)).clamp_min(1.0)
                query_msg = (agent_h * w).sum(dim=(1, 2)) / denom
        return z_agent, z_cand, z_conf, query_msg

    def forward(
        self,
        agent_history: torch.Tensor,
        agent_mask: torch.Tensor,
        candidate_traj: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        conflict_regions: torch.Tensor | None = None,
        conflict_mask: torch.Tensor | None = None,
        ego_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x_agent = self._history_mean(agent_history)
        if ego_mask is None:
            agent_type = torch.full_like(agent_mask.long(), 3)
            if agent_type.shape[1] > 0:
                agent_type[:, 0] = 0
        else:
            agent_type = torch.where(ego_mask.bool(), torch.zeros_like(agent_mask.long()), torch.full_like(agent_mask.long(), 3))
        z_agent = self.agent_proj(x_agent) + self.type_embed(agent_type)
        z_cand = None
        if candidate_traj is not None:
            z_cand = self.candidate_proj(candidate_traj.mean(dim=2)) + self.type_embed(torch.ones(candidate_traj.shape[:2], device=candidate_traj.device, dtype=torch.long))
        z_conf = None
        if conflict_regions is not None:
            z_conf = self.conflict_proj(conflict_regions) + self.type_embed(torch.full(conflict_regions.shape[:2], 2, device=conflict_regions.device, dtype=torch.long))

        z_agent, z_cand, z_conf, query_msg = self._inject_edge_messages(
            z_agent,
            agent_mask,
            x_agent,
            z_cand,
            candidate_traj,
            candidate_mask,
            z_conf,
            conflict_regions,
            conflict_mask,
        )

        pieces = [z_agent]
        masks = [agent_mask.bool()]
        if z_cand is not None:
            pieces.append(z_cand)
            masks.append(candidate_mask.bool() if candidate_mask is not None else torch.ones(z_cand.shape[:2], device=z_cand.device, dtype=torch.bool))
        if z_conf is not None:
            pieces.append(z_conf)
            masks.append(conflict_mask.bool() if conflict_mask is not None else torch.ones(z_conf.shape[:2], device=z_conf.device, dtype=torch.bool))
        if self.use_conflict_query:
            query = self.conflict_query.expand(agent_history.shape[0], 1, -1) + self.type_embed(torch.full((agent_history.shape[0], 1), 3, device=agent_history.device, dtype=torch.long))
            if query_msg is not None:
                query = query + self.query_update(query_msg).unsqueeze(1)
            pieces.append(query)
            masks.append(torch.ones(agent_history.shape[0], 1, device=agent_history.device, dtype=torch.bool))

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
            offset += z_conf.shape[1]
        if self.use_conflict_query:
            out["z_conflict_query"] = enc[:, offset]
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        out["z_graph"] = (enc * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        if self.use_conflict_query and "z_conflict_query" in out:
            out["z_graph"] = 0.5 * (out["z_graph"] + out["z_conflict_query"])
        out["z_ego"] = enc[:, 0]
        return out
