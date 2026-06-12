from __future__ import annotations

import torch
from torch import nn

from cowp.models.candidate_encoder import CandidateEncoder
from cowp.models.graph_encoder import GraphEncoder
from cowp.models.natural_decoder import NaturalDecoder
from cowp.models.planner_head import PlannerHead
from cowp.models.response_decoder import ResponseDecoder
from cowp.models.witness_decoder import WitnessDecoder
from cowp.data.womd_features import build_agent_history_from_womd, has_womd_state


class COWPModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", cfg)
        d_model = int(m.get("d_model", 128))
        self.graph = GraphEncoder(int(m.get("d_state", 11)), d_model, int(m.get("num_heads", 4)), int(m.get("num_layers", 3)), float(m.get("dropout", 0.1)))
        self.candidate_encoder = CandidateEncoder(d_model=d_model, dropout=float(m.get("dropout", 0.1)))
        self.natural_decoder = NaturalDecoder(d_model=d_model, modes=int(m.get("max_natural_alternatives", 24)), future_steps=int(m.get("future_steps", 80)))
        self.response_decoder = ResponseDecoder(d_model=d_model, responses=int(m.get("max_safe_responses", 32)), future_steps=int(m.get("future_steps", 80)))
        self.witness_decoder = WitnessDecoder(d_model=d_model, token_count=int(m.get("token_count", 7)))
        self.planner = PlannerHead(d_model=d_model)
        self.max_agents = int(m.get("max_agents", 128))
        self.history_steps = int(m.get("history_steps", 11))
        self.d_state = int(m.get("d_state", 11))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Prefer real WOMD tf.Example tensors from tensor_cache.  Earlier versions
        # only checked state/history and state/all, so merged tensor caches silently
        # fell back to label-only natural trajectories.
        if "state/history" in batch:
            agent_history = batch["state/history"].float()
            agent_mask = batch.get("state/agent_valid", torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)).bool()
        elif has_womd_state(batch):
            agent_history, agent_mask = build_agent_history_from_womd(
                batch,
                max_agents=self.max_agents,
                history_steps=self.history_steps,
                d_state=self.d_state,
            )
        elif "state/all" in batch:
            all_state = batch["state/all"].float()
            agent_history = all_state[:, :, : min(11, all_state.shape[2]), :11] if all_state.ndim == 4 else all_state[:, :, None, :11]
            agent_mask = batch.get("state/agent_valid", torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)).bool()
        else:
            # Last-resort toy/label-only fallback.  This path is intentionally kept
            # for unit tests and diagnostics, but production training should use
            # tensor_cache with WOMD state features.
            nat = batch["cowp/natural/traj"].float()
            B, A = nat.shape[:2]
            N = max(int(batch["cowp/critical/track_index"].max().item() + 1) if batch["cowp/critical/track_index"].numel() else A, A, 1)
            agent_history = torch.zeros(B, N, 1, self.d_state, device=nat.device)
            agent_mask = torch.zeros(B, N, device=nat.device, dtype=torch.bool)
            for a in range(A):
                idx = batch["cowp/critical/track_index"][:, a].clamp_min(0).long()
                vals = nat[:, a, 0, 0]
                for b in range(B):
                    agent_history[b, idx[b], 0, 0] = vals[b, 0]
                    agent_history[b, idx[b], 0, 1] = vals[b, 1]
                    agent_history[b, idx[b], 0, 6] = vals[b, 2]
                    agent_history[b, idx[b], 0, 3:5] = vals[b, 3:5]
                    agent_history[b, idx[b], 0, 7:9] = vals[b, 5:7]
                    agent_history[b, idx[b], 0, 10] = 1.0
                    agent_mask[b, idx[b]] = True
            agent_mask[:, 0] = True
        cand_traj = batch["cowp/candidates/trajectory"].float()
        cand_mask = batch["cowp/candidates/valid"].bool()
        conflict = batch.get("map/conflict_regions")
        conflict_mask = batch.get("map/conflict_region_valid")
        enc = self.graph(agent_history, agent_mask, cand_traj, cand_mask, conflict.float() if conflict is not None else None, conflict_mask.bool() if conflict_mask is not None else None)
        z_cand = self.candidate_encoder(cand_traj, batch["cowp/candidates/macro_type"].long())
        if "z_candidate_context" in enc:
            z_cand = z_cand + enc["z_candidate_context"]
        critical_idx = batch["cowp/critical/track_index"].long().clamp_min(0)
        natural = self.natural_decoder(enc["z_agent"], critical_idx)
        response = self.response_decoder(enc["z_agent"], z_cand, enc["z_graph"], critical_idx)
        witness = self.witness_decoder(enc["z_agent"], z_cand, enc["z_graph"], critical_idx)
        witness_prob = torch.sigmoid(witness["exist_logits"])
        planner_score = self.planner(z_cand, batch.get("cowp/candidates/ego_utility_prior", torch.zeros_like(cand_mask, dtype=torch.float32)).float(), witness_prob, witness["opr"], batch.get("cowp/candidates/conventional_safe"))
        return {"enc": enc, "natural": natural, "response": response, "witness": witness, "planner_score": planner_score}
